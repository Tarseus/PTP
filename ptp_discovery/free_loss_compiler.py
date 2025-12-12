from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from .free_loss_ir import FreeLossIR, ir_from_json


LossFn = Callable[[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, Any]], torch.Tensor]


class CompileError(Exception):
    pass


@dataclass
class CompiledFreeLoss:
    ir: FreeLossIR
    loss_fn: LossFn


ALLOWED_OPERATORS = {
    "logsigmoid",
    "softplus",
    "sigmoid",
    "exp",
    "log",
    "tanh",
    "relu",
    "clamp",
    "normalize",
    "zscore",
    "rank_gap",
}


def _extract_json_object(text: str) -> Mapping[str, Any]:
    """Extract the first top-level JSON object from a string.

    This is more robust than taking text between the first "{" and the last
    "}", which can easily capture multiple objects or trailing data.
    """

    start = text.find("{")
    if start == -1:
        raise CompileError("No JSON object found in LLM output.")

    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None or end <= start:
        raise CompileError("Failed to locate a complete JSON object in LLM output.")

    snippet = text[start : end + 1]

    # Be tolerant to invalid backslash escapes that sometimes appear in LLM
    # generated JSON (e.g., LaTeX-like `\alpha`). JSON only allows a limited
    # set of escapes after `\`, so we rewrite any other `\x` into `\\x` so
    # that it decodes as a literal backslash.
    invalid_escape_pattern = re.compile(r'\\(?!["\\/bfnrtu])')
    sanitized_snippet = invalid_escape_pattern.sub(r"\\\\", snippet)

    try:
        return json.loads(sanitized_snippet)
    except json.JSONDecodeError as exc:
        raise CompileError(f"Failed to parse JSON from LLM output: {exc}") from exc


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return x / (std + eps)


def _safe_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def _rank_gap(cost_a: torch.Tensor, cost_b: torch.Tensor) -> torch.Tensor:
    return cost_b - cost_a


def _build_operator_table() -> Dict[str, Callable[..., torch.Tensor]]:
    return {
        "logsigmoid": F.logsigmoid,
        "softplus": F.softplus,
        "sigmoid": torch.sigmoid,
        "exp": torch.exp,
        "log": torch.log,
        "tanh": torch.tanh,
        "relu": F.relu,
        "clamp": lambda x, min=-10.0, max=10.0: torch.clamp(x, min=min, max=max),
        "normalize": _safe_normalize,
        "zscore": _safe_zscore,
        "rank_gap": _rank_gap,
    }


def parse_free_loss_from_text(text: str) -> FreeLossIR:
    obj = _extract_json_object(text)
    return ir_from_json(obj)


def compile_free_loss(ir: FreeLossIR, *, operator_whitelist: Sequence[str] | None = None) -> CompiledFreeLoss:
    ops = set(ir.operators_used)
    if operator_whitelist is None:
        operator_whitelist = ALLOWED_OPERATORS
    allowed = set(operator_whitelist)

    if not ops.issubset(allowed):
        unknown = sorted(ops - allowed)
        raise CompileError(f"operators_used contains non-whitelisted operators: {unknown}")

    # Prefer a concrete Python implementation provided directly by the LLM
    # in ir.code. This avoids a second model call during compilation and
    # makes the search operate directly over executable loss functions.
    code_str = (ir.code or "").strip()
    if code_str:
        local_ns: Dict[str, Any] = {"torch": torch, "F": F}
        global_ns: Dict[str, Any] = {}
        try:
            exec(code_str, local_ns, global_ns)
        except Exception as exc:  # noqa: BLE001
            raise CompileError(f"Failed to exec loss code from IR: {exc}") from exc

        fn = global_ns.get("generated_loss") or local_ns.get("generated_loss")
        if not callable(fn):
            raise CompileError(
                "Loss code did not define a callable 'generated_loss(batch, model_output, extra)'."
            )

        def loss_fn(
            batch: Mapping[str, Any],
            model_output: Mapping[str, torch.Tensor],
            extra: Mapping[str, Any],
        ) -> torch.Tensor:
            return fn(batch, model_output, extra)
    else:
        # Backward-compatible fallback: use a simple template-based loss
        # when no explicit code is provided in the IR.
        table = _build_operator_table()

        def loss_fn(
            batch: Mapping[str, Any],
            model_output: Mapping[str, torch.Tensor],
            extra: Mapping[str, Any],
        ) -> torch.Tensor:
            pair_cost_a = batch["cost_a"]
            pair_cost_b = batch["cost_b"]
            logit_diff = batch.get("logit_diff")
            if logit_diff is None:
                log_prob_w = batch.get("log_prob_w")
                log_prob_l = batch.get("log_prob_l")
                if log_prob_w is None or log_prob_l is None:
                    raise RuntimeError("batch must provide either logit_diff or log_prob_w/log_prob_l")
                alpha = float(ir.hyperparams.get("alpha", extra.get("alpha", 1.0)))
                logit_diff = alpha * (log_prob_w - log_prob_l)

            cost_gap = _rank_gap(pair_cost_a, pair_cost_b)
            cost_gap_z = _safe_zscore(cost_gap)

            scale = float(ir.hyperparams.get("scale", 1.0))
            margin = float(ir.hyperparams.get("margin", 0.0))

            x = scale * (logit_diff - margin * cost_gap_z)
            loss = -table["logsigmoid"](x)

            weight = batch.get("weight")
            if weight is not None:
                loss = loss * weight

            return loss.mean()

    return CompiledFreeLoss(ir=ir, loss_fn=loss_fn)
