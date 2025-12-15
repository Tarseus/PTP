from __future__ import annotations

import ast
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

    # Additionally, some models may emit raw control characters (e.g., literal
    # newlines or tabs) inside JSON strings without escaping them, which is
    # invalid JSON. We conservatively post-process the snippet to escape such
    # characters only when they appear inside string literals.
    def _escape_control_chars_in_strings(s: str) -> str:
        out_chars: list[str] = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                out_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                out_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                out_chars.append(ch)
                in_string = not in_string
                continue
            if in_string and ch in ("\n", "\r", "\t"):
                # Replace raw control characters with escaped counterparts.
                if ch == "\n":
                    out_chars.append("\\n")
                elif ch == "\r":
                    out_chars.append("\\r")
                elif ch == "\t":
                    out_chars.append("\\t")
                continue
            out_chars.append(ch)
        return "".join(out_chars)

    sanitized_snippet = _escape_control_chars_in_strings(sanitized_snippet)

    try:
        return json.loads(sanitized_snippet)
    except json.JSONDecodeError as exc:
        raise CompileError(f"Failed to parse JSON from LLM output: {exc}") from exc


class _SafeCodeValidator(ast.NodeVisitor):
    """Best-effort static validation for user-provided loss code.

    The goal is to rule out obviously dangerous constructs (imports, exec,
    file I/O, process control, etc.) before executing the code in a tightly
    restricted environment.
    """

    _FORBIDDEN_CALL_NAMES = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }

    _FORBIDDEN_ATTR_BASES = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
    }

    def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
        raise CompileError("Loss code must not use import statements.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # type: ignore[override]
        raise CompileError("Loss code must not use import-from statements.")

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func = node.func
        if isinstance(func, ast.Name) and func.id in self._FORBIDDEN_CALL_NAMES:
            raise CompileError(f"Loss code calls forbidden function '{func.id}'.")
        if isinstance(func, ast.Attribute):
            # Disallow e.g. os.system, sys.exit, subprocess.Popen, etc.
            base = func.value
            if isinstance(base, ast.Name) and base.id in self._FORBIDDEN_ATTR_BASES:
                raise CompileError(
                    f"Loss code must not access '{base.id}.{func.attr}'. "
                    "Only tensor-level math using torch/F is allowed."
                )
        self.generic_visit(node)


def _validate_user_code(code_str: str) -> None:
    """Run lightweight static checks on user-provided loss code."""

    try:
        tree = ast.parse(code_str, mode="exec")
    except SyntaxError as exc:
        raise CompileError(f"Loss code has syntax error: {exc}") from exc

    validator = _SafeCodeValidator()
    validator.visit(tree)


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
        _validate_user_code(code_str)

        # Execute in a tightly restricted namespace. We deliberately strip
        # builtins to avoid access to filesystem, subprocesses, etc.
        safe_globals: Dict[str, Any] = {
            "__builtins__": {},
            "torch": torch,
            "F": F,
        }
        local_ns: Dict[str, Any] = {}
        try:
            exec(code_str, safe_globals, local_ns)
        except Exception as exc:  # noqa: BLE001
            raise CompileError(f"Failed to exec loss code from IR: {exc}") from exc

        fn = local_ns.get("generated_loss")
        if not callable(fn):
            raise CompileError(
                "Loss code did not define a callable 'generated_loss(batch, model_output, extra)'."
            )

        def loss_fn(
            batch: Mapping[str, Any],
            model_output: Mapping[str, torch.Tensor],
            extra: Mapping[str, Any] | None,
        ) -> torch.Tensor:
            merged_extra: Dict[str, Any] = {"hyperparams": ir.hyperparams}
            if extra:
                merged_extra.update(extra)
            return fn(batch, model_output, merged_extra)
    else:
        # Backward-compatible fallback: use a simple template-based loss
        # when no explicit code is provided in the IR.
        table = _build_operator_table()

        def _resolve_hparam(raw: Any, default: float) -> float:
            if isinstance(raw, dict):
                raw = raw.get("value", default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(default)

        def loss_fn(
            batch: Mapping[str, Any],
            model_output: Mapping[str, torch.Tensor],
            extra: Mapping[str, Any] | None,
        ) -> torch.Tensor:
            pair_cost_a = batch["cost_a"]
            pair_cost_b = batch["cost_b"]
            logit_diff = batch.get("logit_diff")
            if logit_diff is None:
                log_prob_w = batch.get("log_prob_w")
                log_prob_l = batch.get("log_prob_l")
                if log_prob_w is None or log_prob_l is None:
                    raise RuntimeError("batch must provide either logit_diff or log_prob_w/log_prob_l")
                extra = extra or {}
                alpha_cfg = ir.hyperparams.get("alpha", extra.get("alpha", 1.0))
                alpha = _resolve_hparam(alpha_cfg, 1.0)
                logit_diff = alpha * (log_prob_w - log_prob_l)

            cost_gap = _rank_gap(pair_cost_a, pair_cost_b)
            cost_gap_z = _safe_zscore(cost_gap)

            extra = extra or {}
            scale_cfg = ir.hyperparams.get("scale", extra.get("scale", 1.0))
            margin_cfg = ir.hyperparams.get("margin", extra.get("margin", 0.0))
            scale = _resolve_hparam(scale_cfg, 1.0)
            margin = _resolve_hparam(margin_cfg, 0.0)

            x = scale * (logit_diff - margin * cost_gap_z)
            loss = -table["logsigmoid"](x)

            weight = batch.get("weight")
            if weight is not None:
                loss = loss * weight

            return loss.mean()

    return CompiledFreeLoss(ir=ir, loss_fn=loss_fn)
