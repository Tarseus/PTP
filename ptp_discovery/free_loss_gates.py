from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import torch

from .free_loss_compiler import CompiledFreeLoss
from .free_loss_ir import FreeLossIR
from fitness.co_features import build_model_output


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


_PAIRWISE_SUPPORTED_KEYS: Set[str] = {
    "log_prob_w",
    "log_prob_l",
    "cost_a",
    "cost_b",
    "delta_z",
    "delta_rank",
    "delta_regret",
    "weight",
}

_SETWISE_SUPPORTED_KEYS: Set[str] = {
    "objective",
    "log_prob",
    "obj_z",
    "rank",
    "regret",
}

_PAIRWISE_REQUIRED_LOGPROB: Set[str] = {"log_prob_w", "log_prob_l"}
_PAIRWISE_REQUIRED_OBJECTIVE_SIGNAL: Set[str] = {
    "delta_z",
    "delta_rank",
    "delta_regret",
    "cost_a",
    "cost_b",
}

_SETWISE_REQUIRED_LOGPROB: Set[str] = {"log_prob"}
_SETWISE_REQUIRED_OBJECTIVE_SIGNAL: Set[str] = {"obj_z", "rank", "regret", "objective"}


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
                "implementation_hint.expects must be a non-empty list of input keys."
            ),
        )

    expects_set = {str(x) for x in expects}
    mode = str(getattr(ir.implementation_hint, "mode", "pairwise") or "pairwise").strip().lower()
    if mode not in {"pairwise", "setwise"}:
        return StaticGateResult(ok=False, reason=f"implementation_hint.mode invalid: {mode}")

    if not (ir.code or "").strip():
        # Template-based losses require the legacy pairwise inputs.
        if mode != "pairwise":
            return StaticGateResult(ok=False, reason="template_loss_requires_pairwise_mode")
        required = {"cost_a", "cost_b", "log_prob_w", "log_prob_l"}
        missing = required - expects_set
        extra = expects_set - (_PAIRWISE_SUPPORTED_KEYS | {"logit_diff"})
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
        return StaticGateResult(ok=True)

    if mode == "pairwise":
        missing_lp = _PAIRWISE_REQUIRED_LOGPROB - expects_set
        if missing_lp:
            return StaticGateResult(
                ok=False,
                reason=f"implementation_hint.expects missing required keys: {sorted(missing_lp)}",
            )
        if not (expects_set & _PAIRWISE_REQUIRED_OBJECTIVE_SIGNAL):
            return StaticGateResult(
                ok=False,
                reason=(
                    "implementation_hint.expects must include at least one objective signal "
                    "from {delta_z, delta_rank, delta_regret, cost_a, cost_b}."
                ),
            )
        extra = expects_set - _PAIRWISE_SUPPORTED_KEYS
        if extra:
            return StaticGateResult(
                ok=False,
                reason=f"implementation_hint.expects contains unsupported keys: {sorted(extra)}",
            )
    else:
        missing_lp = _SETWISE_REQUIRED_LOGPROB - expects_set
        if missing_lp:
            return StaticGateResult(
                ok=False,
                reason=f"implementation_hint.expects missing required keys: {sorted(missing_lp)}",
            )
        if not (expects_set & _SETWISE_REQUIRED_OBJECTIVE_SIGNAL):
            return StaticGateResult(
                ok=False,
                reason=(
                    "implementation_hint.expects must include at least one objective signal "
                    "from {obj_z, rank, regret, objective}."
                ),
            )
        extra = expects_set - _SETWISE_SUPPORTED_KEYS
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
    model_output: Mapping[str, torch.Tensor] | None = None,
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

    dummy_output: Dict[str, torch.Tensor] = dict(model_output or {})

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

    mode = str(getattr(compiled.ir.implementation_hint, "mode", "pairwise") or "pairwise").strip().lower()
    expects = [str(x) for x in (compiled.ir.implementation_hint.expects or [])]
    expects_set = set(expects)
    if mode != "pairwise":
        return PreferenceSemanticGateResult(ok=True, reason="skipped_non_pairwise")

    for _ in range(trials):
        # Base synthetic batch.
        log_prob_l = (torch.rand(batch_size) * -20.0).requires_grad_(True)
        log_prob_w = (
            log_prob_l + torch.empty(batch_size).uniform_(-5.0, 5.0)
        ).requires_grad_(True)

        cost_a = torch.rand(batch_size)
        gap = torch.rand(batch_size)
        cost_b = cost_a + gap

        delta_z = gap * 2.0
        delta_rank = (gap > 0.5).to(dtype=torch.float32)
        delta_regret = gap / (gap.median() + 1e-6)

        full_batch: Dict[str, torch.Tensor] = {
            "log_prob_w": log_prob_w,
            "log_prob_l": log_prob_l,
            "cost_a": cost_a,
            "cost_b": cost_b,
            "delta_z": delta_z,
            "delta_rank": delta_rank,
            "delta_regret": delta_regret,
            "weight": torch.ones(batch_size),
        }
        batch = {k: full_batch[k] for k in expects if k in full_batch}

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
        full_swap: Dict[str, torch.Tensor] = {
            "log_prob_w": log_prob_l.detach(),
            "log_prob_l": log_prob_w.detach(),
            "cost_a": cost_b,
            "cost_b": cost_a,
            "delta_z": -delta_z,
            "delta_rank": -delta_rank,
            "delta_regret": -delta_regret,
            "weight": torch.ones(batch_size),
        }
        swap_batch = {k: full_swap[k] for k in expects if k in full_swap}
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

        log_prob_l2 = (torch.rand(batch_size) * -20.0).requires_grad_(True)
        log_prob_w2 = (
            log_prob_l2 + torch.empty(batch_size).uniform_(-5.0, 5.0)
        ).requires_grad_(True)

        cost_b_small = cost_a + small_gap
        cost_b_large = cost_a + large_gap

        full_small: Dict[str, torch.Tensor] = {
            "log_prob_w": log_prob_w2,
            "log_prob_l": log_prob_l2,
            "cost_a": cost_a,
            "cost_b": cost_b_small,
            "delta_z": small_gap * 2.0,
            "delta_rank": (small_gap > 0.05).to(dtype=torch.float32),
            "delta_regret": small_gap / (small_gap.median() + 1e-6),
            "weight": torch.ones(batch_size),
        }
        full_large: Dict[str, torch.Tensor] = {
            "log_prob_w": log_prob_w2,
            "log_prob_l": log_prob_l2,
            "cost_a": cost_a,
            "cost_b": cost_b_large,
            "delta_z": large_gap * 2.0,
            "delta_rank": (large_gap > 0.75).to(dtype=torch.float32),
            "delta_regret": large_gap / (large_gap.median() + 1e-6),
            "weight": torch.ones(batch_size),
        }
        batch_small = {k: full_small[k] for k in expects if k in full_small}
        batch_large = {k: full_large[k] for k in expects if k in full_large}
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


@dataclass
class ObjectiveSensitivityGateResult:
    ok: bool
    reason: str = ""
    abs_delta: float | None = None
    rel_delta: float | None = None


@dataclass
class AffineInvarianceGateResult:
    ok: bool
    reason: str = ""
    abs_delta: float | None = None
    rel_delta: float | None = None


def _loss_value(
    compiled: CompiledFreeLoss,
    *,
    batch: Mapping[str, Any],
    model_output: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return compiled.loss_fn(batch=batch, model_output=model_output, extra={})


def _delta_metrics(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[float, float]:
    abs_delta = float((a - b).abs().item())
    denom = float((a.abs() + b.abs() + eps).item())
    rel_delta = abs_delta / denom
    return abs_delta, rel_delta


def run_objective_sensitivity_gate(
    compiled: CompiledFreeLoss,
    *,
    min_abs_delta: float = 1e-3,
    min_rel_delta: float = 1e-2,
    pairwise_batch_size: int = 64,
    setwise_batch_size: int = 8,
    setwise_pomo: int = 16,
) -> ObjectiveSensitivityGateResult:
    """Require the loss to respond to changes in objective-derived inputs."""

    mode = str(getattr(compiled.ir.implementation_hint, "mode", "pairwise") or "pairwise").strip().lower()
    expects = [str(x) for x in (compiled.ir.implementation_hint.expects or [])]
    expects_set = set(expects)

    try:
        if mode == "pairwise":
            log_prob_l = (torch.rand(pairwise_batch_size) * -8.0).requires_grad_(False)
            log_prob_w = (log_prob_l + torch.empty(pairwise_batch_size).uniform_(-3.0, 3.0)).requires_grad_(
                False
            )

            cost_a = torch.rand(pairwise_batch_size)
            gap = torch.rand(pairwise_batch_size)
            cost_b = cost_a + gap

            obj_small = {
                "delta_z": gap * 0.1,
                "delta_rank": (gap > 0.5).to(dtype=torch.float32) * 0.1,
                "delta_regret": gap * 0.1,
                "cost_a": cost_a,
                "cost_b": cost_b,
            }
            obj_large = {
                "delta_z": gap * 5.0,
                "delta_rank": (gap > 0.5).to(dtype=torch.float32) * 5.0,
                "delta_regret": gap * 5.0,
                "cost_a": cost_a * 5.0,
                "cost_b": cost_b * 5.0,
            }

            full_1: Dict[str, Any] = {
                "log_prob_w": log_prob_w,
                "log_prob_l": log_prob_l,
                **obj_small,
                "weight": torch.ones(pairwise_batch_size),
            }
            full_2: Dict[str, Any] = {
                "log_prob_w": log_prob_w,
                "log_prob_l": log_prob_l,
                **obj_large,
                "weight": torch.ones(pairwise_batch_size),
            }
            batch_1 = {k: full_1[k] for k in expects if k in full_1}
            batch_2 = {k: full_2[k] for k in expects if k in full_2}
            loss_1 = _loss_value(compiled, batch=batch_1, model_output={})
            loss_2 = _loss_value(compiled, batch=batch_2, model_output={})
        else:
            objective = torch.rand(setwise_batch_size, setwise_pomo)
            # Ensure non-affine perturbations to trigger changes in invariant features.
            noise = torch.randn_like(objective) * 0.5
            objective_2 = (objective + noise).clamp_min(0.0)
            log_prob = torch.rand(setwise_batch_size, setwise_pomo) * -8.0

            model_out_1, _ = build_model_output(objective=objective, log_prob=log_prob)
            model_out_2, _ = build_model_output(objective=objective_2, log_prob=log_prob)
            out_1 = {k: model_out_1[k] for k in expects if k in model_out_1}
            out_2 = {k: model_out_2[k] for k in expects if k in model_out_2}
            loss_1 = _loss_value(compiled, batch={}, model_output=out_1)
            loss_2 = _loss_value(compiled, batch={}, model_output=out_2)

        if not torch.isfinite(loss_1) or not torch.isfinite(loss_2):
            return ObjectiveSensitivityGateResult(ok=False, reason="non_finite_loss")

        abs_delta, rel_delta = _delta_metrics(loss_1, loss_2)
        ok = abs_delta >= float(min_abs_delta) or rel_delta >= float(min_rel_delta)
        return ObjectiveSensitivityGateResult(
            ok=ok,
            reason="ok" if ok else "insensitive_to_objective",
            abs_delta=abs_delta,
            rel_delta=rel_delta,
        )
    except KeyError as exc:
        return ObjectiveSensitivityGateResult(ok=False, reason=f"missing_key: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ObjectiveSensitivityGateResult(ok=False, reason=f"error: {exc}")


def run_affine_invariance_gate(
    compiled: CompiledFreeLoss,
    *,
    max_abs_delta: float = 1e-3,
    max_rel_delta: float = 1e-2,
    pairwise_batch_size: int = 64,
    setwise_batch_size: int = 8,
    setwise_pomo: int = 16,
    a: float = 7.0,
    b: float = 3.0,
) -> AffineInvarianceGateResult:
    """Require approximate invariance under objective' = a*objective + b."""

    mode = str(getattr(compiled.ir.implementation_hint, "mode", "pairwise") or "pairwise").strip().lower()
    expects = [str(x) for x in (compiled.ir.implementation_hint.expects or [])]
    expects_set = set(expects)

    try:
        if mode == "pairwise":
            log_prob_l = (torch.rand(pairwise_batch_size) * -8.0).requires_grad_(False)
            log_prob_w = (log_prob_l + torch.empty(pairwise_batch_size).uniform_(-3.0, 3.0)).requires_grad_(
                False
            )

            cost_a = torch.rand(pairwise_batch_size)
            gap = torch.rand(pairwise_batch_size)
            cost_b = cost_a + gap

            # Invariant deltas (as if recomputed from affine-transformed objectives).
            delta_z = gap * 2.0
            delta_rank = (gap > 0.5).to(dtype=torch.float32)
            delta_regret = gap / (gap.median() + 1e-6)

            full_1: Dict[str, Any] = {
                "log_prob_w": log_prob_w,
                "log_prob_l": log_prob_l,
                "cost_a": cost_a,
                "cost_b": cost_b,
                "delta_z": delta_z,
                "delta_rank": delta_rank,
                "delta_regret": delta_regret,
                "weight": torch.ones(pairwise_batch_size),
            }
            full_2: Dict[str, Any] = dict(full_1)
            if "cost_a" in expects_set:
                full_2["cost_a"] = a * cost_a + b
            if "cost_b" in expects_set:
                full_2["cost_b"] = a * cost_b + b

            batch_1 = {k: full_1[k] for k in expects if k in full_1}
            batch_2 = {k: full_2[k] for k in expects if k in full_2}
            loss_1 = _loss_value(compiled, batch=batch_1, model_output={})
            loss_2 = _loss_value(compiled, batch=batch_2, model_output={})
        else:
            objective = torch.rand(setwise_batch_size, setwise_pomo)
            log_prob = torch.rand(setwise_batch_size, setwise_pomo) * -8.0
            objective_2 = a * objective + b

            model_out_1, _ = build_model_output(objective=objective, log_prob=log_prob)
            model_out_2, _ = build_model_output(objective=objective_2, log_prob=log_prob)
            out_1 = {k: model_out_1[k] for k in expects if k in model_out_1}
            out_2 = {k: model_out_2[k] for k in expects if k in model_out_2}
            loss_1 = _loss_value(compiled, batch={}, model_output=out_1)
            loss_2 = _loss_value(compiled, batch={}, model_output=out_2)

        if not torch.isfinite(loss_1) or not torch.isfinite(loss_2):
            return AffineInvarianceGateResult(ok=False, reason="non_finite_loss")

        abs_delta, rel_delta = _delta_metrics(loss_1, loss_2)
        ok = abs_delta <= float(max_abs_delta) or rel_delta <= float(max_rel_delta)
        return AffineInvarianceGateResult(
            ok=ok,
            reason="ok" if ok else "affine_invariance_violation",
            abs_delta=abs_delta,
            rel_delta=rel_delta,
        )
    except KeyError as exc:
        return AffineInvarianceGateResult(ok=False, reason=f"missing_key: {exc}")
    except Exception as exc:  # noqa: BLE001
        return AffineInvarianceGateResult(ok=False, reason=f"error: {exc}")
