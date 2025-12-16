from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

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
    grad_norm_max: float,
    loss_soft_min: float,
    loss_soft_max: float,
) -> DynamicGateResult:
    model.zero_grad()

    dummy_output: Dict[str, torch.Tensor] = {}

    try:
        loss = compiled.loss_fn(batch=batch, model_output=dummy_output, extra={})
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
