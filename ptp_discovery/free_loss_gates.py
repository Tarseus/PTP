from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Set

import torch

from .free_loss_compiler import CompiledFreeLoss
from .free_loss_ir import FreeLossIR


@dataclass
class StaticGateResult:
    ok: bool
    reason: str = ""


@dataclass
class DynamicGateResult:
    ok: bool
    reason: str = ""
    loss_value: float | None = None
    grad_norm: float | None = None


@dataclass
class PreferenceSemanticGateResult:
    ok: bool
    reason: str = ""
    mono_pass_rate: float | None = None
    swap_pass_rate: float | None = None
    gap_pass_rate: float | None = None


_REQUIRED_BATCH_KEYS: Set[str] = {
    "cost_a",
    "cost_b",
    "log_prob_w",
    "log_prob_l",
}


def run_static_gates(
    ir: FreeLossIR,
    *,
    operator_whitelist: Sequence[str],
) -> StaticGateResult:
    if not ir.name:
        return StaticGateResult(ok=False, reason="Missing name.")
    if not ir.pseudocode:
        return StaticGateResult(ok=False, reason="Missing pseudocode.")
    # We keep operators_used as a descriptive field but no longer enforce
    # a hard whitelist. This allows the discovery process to explore more
    # freely; safety is enforced at the code level and via dynamic gates.
    if not ir.operators_used:
        return StaticGateResult(ok=False, reason="operators_used must be non-empty.")

    returns_str = (ir.implementation_hint.returns or "").strip().lower()
    # Be tolerant to descriptive strings like "a scalar loss value ...".
    if returns_str and "scalar" not in returns_str:
        return StaticGateResult(
            ok=False,
            reason="implementation_hint.returns must describe a scalar output.",
        )

    expects = ir.implementation_hint.expects or []
    if not isinstance(expects, (list, tuple)) or not expects:
        return StaticGateResult(
            ok=False,
            reason=(
                "implementation_hint.expects must list "
                "cost_a/cost_b/log_prob_w/log_prob_l."
            ),
        )

    expects_set = {str(x) for x in expects}
    missing = _REQUIRED_BATCH_KEYS - expects_set
    extra = expects_set - _REQUIRED_BATCH_KEYS
    if missing:
        return StaticGateResult(
            ok=False,
            reason=f"implementation_hint.expects missing required keys: {sorted(missing)}",
        )
    if extra:
        return StaticGateResult(
            ok=False,
            reason=f"implementation_hint.expects contains unsupported keys: {sorted(extra)}",
        )

    for key, value in ir.hyperparams.items():
        if isinstance(value, (int, float)):
            if not torch.isfinite(torch.tensor(float(value))):
                return StaticGateResult(ok=False, reason=f"hyperparameter {key} is non-finite.")

    return StaticGateResult(ok=True)


def run_dynamic_gates(
    compiled: CompiledFreeLoss,
    batch: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    required_batch_keys: Sequence[str] | None = None,
    grad_norm_max: float,
    loss_soft_min: float,
    loss_soft_max: float,
) -> DynamicGateResult:
    required_keys = set(required_batch_keys or [])
    if required_keys:
        expects = set(compiled.ir.implementation_hint.expects or [])
        if expects != required_keys:
            missing = required_keys - expects
            extra = expects - required_keys
            reason_parts = []
            if missing:
                reason_parts.append(f"missing_expects={sorted(missing)}")
            if extra:
                reason_parts.append(f"unsupported_expects={sorted(extra)}")
            reason = "invalid_expects"
            if reason_parts:
                reason = f"{reason}: " + ", ".join(reason_parts)
            return DynamicGateResult(ok=False, reason=reason)

        batch_keys = set(batch.keys())
        missing_batch = required_keys - batch_keys
        extra_batch = batch_keys - required_keys
        if missing_batch:
            return DynamicGateResult(
                ok=False,
                reason=f"missing_batch_key: {sorted(missing_batch)}",
            )
        if extra_batch:
            return DynamicGateResult(
                ok=False,
                reason=f"extra_batch_key: {sorted(extra_batch)}",
            )

    model.zero_grad()

    dummy_output: Dict[str, torch.Tensor] = {}

    try:
        loss = compiled.loss_fn(batch=batch, model_output=dummy_output, extra={})
        if not isinstance(loss, torch.Tensor):
            return DynamicGateResult(
                ok=False,
                reason=f"loss_not_tensor: {type(loss)}",
            )
    except KeyError as exc:
        return DynamicGateResult(ok=False, reason=f"missing_batch_key: {exc}")
    except NameError as exc:
        return DynamicGateResult(ok=False, reason=f"missing_dependency: {exc}")
    except Exception as exc:  # noqa: BLE001
        return DynamicGateResult(ok=False, reason=f"forward_error: {exc}")

    if not torch.isfinite(loss):
        return DynamicGateResult(ok=False, reason="loss is not finite.")

    # Some candidate losses may not depend on model parameters for this
    # synthetic batch, in which case loss.requires_grad will be False.
    # In that situation calling backward() would raise an error, but
    # we still want to allow the candidate as long as the loss value
    # itself is well-behaved. We therefore only run a backward pass and
    # gradient-norm check when gradients are actually defined.
    grad_norm = 0.0
    if loss.requires_grad:
        try:
            loss.backward()
        except Exception as exc:  # noqa: BLE001
            return DynamicGateResult(ok=False, reason=f"backward_error: {exc}")

        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return DynamicGateResult(ok=False, reason="NaN/Inf in gradients.")
            total_norm_sq += float(p.grad.norm().item() ** 2)
        grad_norm = total_norm_sq ** 0.5

    if grad_norm > grad_norm_max:
        return DynamicGateResult(
            ok=False,
            reason=f"grad_norm {grad_norm:.4f} exceeds max {grad_norm_max:.4f}",
            loss_value=float(loss.item()),
            grad_norm=grad_norm,
        )

    loss_val = float(loss.item())
    if loss_val < loss_soft_min or loss_val > loss_soft_max:
        return DynamicGateResult(
            ok=False,
            reason=f"loss {loss_val:.4f} outside soft range [{loss_soft_min}, {loss_soft_max}]",
            loss_value=loss_val,
            grad_norm=grad_norm,
        )

    return DynamicGateResult(
        ok=True,
        reason="ok",
        loss_value=loss_val,
        grad_norm=grad_norm,
    )


def run_preference_semantic_gates(
    compiled: CompiledFreeLoss,
    *,
    trials: int = 6,
    batch_size: int = 128,
    min_pass_rate: float = 0.8,
    swap_tolerance: float = 1e-3,
    gap_min_ratio: float = 0.9,
) -> PreferenceSemanticGateResult:
    """Check preference semantics on synthetic batches.

    We validate that increasing log_prob_w tends to decrease the loss and
    increasing log_prob_l tends to increase it. We also check that swapping
    winner/loser typically increases the loss, and that larger cost gaps
    do not weaken the gradient signal on average.
    """

    if trials <= 0 or batch_size <= 0:
        return PreferenceSemanticGateResult(ok=True, reason="skipped")

    mono_total = 0
    mono_ok = 0
    swap_total = 0
    swap_ok = 0
    gap_total = 0
    gap_ok = 0

    for _ in range(trials):
        # Base synthetic batch.
        cost_a = torch.rand(batch_size)
        gap = torch.rand(batch_size)
        cost_b = cost_a + gap

        log_prob_l = (torch.rand(batch_size) * -20.0).requires_grad_(True)
        log_prob_w = (log_prob_l + torch.empty(batch_size).uniform_(-5.0, 5.0)).requires_grad_(True)

        batch = {
            "cost_a": cost_a,
            "cost_b": cost_b,
            "log_prob_w": log_prob_w,
            "log_prob_l": log_prob_l,
        }

        try:
            loss = compiled.loss_fn(batch=batch, model_output={}, extra={})
            if not isinstance(loss, torch.Tensor):
                return PreferenceSemanticGateResult(
                    ok=False,
                    reason=f"pref_loss_not_tensor: {type(loss)}",
                )
        except Exception as exc:  # noqa: BLE001
            return PreferenceSemanticGateResult(ok=False, reason=f"pref_forward_error: {exc}")

        if not torch.isfinite(loss):
            return PreferenceSemanticGateResult(ok=False, reason="pref_loss_not_finite")

        grad_w, grad_l = torch.autograd.grad(
            loss,
            [log_prob_w, log_prob_l],
            allow_unused=True,
        )
        if grad_w is None or grad_l is None:
            return PreferenceSemanticGateResult(ok=False, reason="pref_grad_missing")

        mono_total += batch_size * 2
        mono_ok += int((grad_w <= 1e-6).sum().item())
        mono_ok += int((grad_l >= -1e-6).sum().item())

        # Swap winner/loser inputs and compare mean loss.
        swap_batch = {
            "cost_a": cost_b,
            "cost_b": cost_a,
            "log_prob_w": log_prob_l.detach(),
            "log_prob_l": log_prob_w.detach(),
        }
        try:
            swap_loss = compiled.loss_fn(batch=swap_batch, model_output={}, extra={})
            if not isinstance(swap_loss, torch.Tensor):
                return PreferenceSemanticGateResult(
                    ok=False,
                    reason=f"pref_loss_not_tensor: {type(swap_loss)}",
                )
        except Exception as exc:  # noqa: BLE001
            return PreferenceSemanticGateResult(ok=False, reason=f"pref_swap_error: {exc}")
        swap_total += 1
        if torch.isfinite(swap_loss) and (swap_loss.item() + swap_tolerance >= loss.item()):
            swap_ok += 1

        # Gap response: larger gaps should not reduce gradient magnitude on average.
        small_gap = torch.rand(batch_size) * 0.1
        large_gap = torch.rand(batch_size) * 1.0 + 0.5
        cost_b_small = cost_a + small_gap
        cost_b_large = cost_a + large_gap

        log_prob_l2 = (torch.rand(batch_size) * -20.0).requires_grad_(True)
        log_prob_w2 = (log_prob_l2 + torch.empty(batch_size).uniform_(-5.0, 5.0)).requires_grad_(True)

        batch_small = {
            "cost_a": cost_a,
            "cost_b": cost_b_small,
            "log_prob_w": log_prob_w2,
            "log_prob_l": log_prob_l2,
        }
        batch_large = {
            "cost_a": cost_a,
            "cost_b": cost_b_large,
            "log_prob_w": log_prob_w2,
            "log_prob_l": log_prob_l2,
        }
        try:
            loss_small = compiled.loss_fn(batch=batch_small, model_output={}, extra={})
            loss_large = compiled.loss_fn(batch=batch_large, model_output={}, extra={})
            if not isinstance(loss_small, torch.Tensor) or not isinstance(loss_large, torch.Tensor):
                return PreferenceSemanticGateResult(
                    ok=False,
                    reason="pref_loss_not_tensor",
                )
        except Exception as exc:  # noqa: BLE001
            return PreferenceSemanticGateResult(ok=False, reason=f"pref_gap_error: {exc}")

        grad_small = torch.autograd.grad(
            loss_small,
            [log_prob_w2, log_prob_l2],
            allow_unused=True,
            retain_graph=True,
        )
        grad_large = torch.autograd.grad(
            loss_large,
            [log_prob_w2, log_prob_l2],
            allow_unused=True,
        )
        if grad_small[0] is None or grad_small[1] is None:
            return PreferenceSemanticGateResult(ok=False, reason="pref_gap_grad_missing")
        if grad_large[0] is None or grad_large[1] is None:
            return PreferenceSemanticGateResult(ok=False, reason="pref_gap_grad_missing")

        grad_delta_small = (grad_small[0] - grad_small[1]).abs().mean().item()
        grad_delta_large = (grad_large[0] - grad_large[1]).abs().mean().item()
        gap_total += 1
        if grad_delta_large + 1e-8 >= gap_min_ratio * grad_delta_small:
            gap_ok += 1

    mono_pass_rate = mono_ok / float(mono_total) if mono_total else 1.0
    swap_pass_rate = swap_ok / float(swap_total) if swap_total else 1.0
    gap_pass_rate = gap_ok / float(gap_total) if gap_total else 1.0

    ok = (
        mono_pass_rate >= min_pass_rate
        and swap_pass_rate >= min_pass_rate
        and gap_pass_rate >= min_pass_rate
    )
    reason = "ok" if ok else "pref_semantic_violation"

    return PreferenceSemanticGateResult(
        ok=ok,
        reason=reason,
        mono_pass_rate=mono_pass_rate,
        swap_pass_rate=swap_pass_rate,
        gap_pass_rate=gap_pass_rate,
    )
