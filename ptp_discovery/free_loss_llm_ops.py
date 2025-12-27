from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from .free_loss_compiler import (
    CompiledFreeLoss,
    compile_free_loss,
    parse_free_loss_from_text,
)
from .free_loss_ir import FreeLossIR


def _load_env() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for free-loss discovery.")


def _make_openai_client() -> OpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _read_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json_object(text: str) -> str:
    """Extract the first complete top-level JSON object from model output.

    More robust than slicing from first '{' to last '}' because LLMs may emit
    multiple objects or trailing text.
    """

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

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
        raise ValueError("Failed to locate a complete JSON object in model output.")

    snippet = text[start : end + 1]

    invalid_escape_pattern = re.compile(r'\\(?!["\\/bfnrtu])')
    sanitized = invalid_escape_pattern.sub(r"\\\\", snippet)

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
                if ch == "\n":
                    out_chars.append("\\n")
                elif ch == "\r":
                    out_chars.append("\\r")
                else:
                    out_chars.append("\\t")
                continue
            out_chars.append(ch)
        return "".join(out_chars)

    return _escape_control_chars_in_strings(sanitized)


def _call_llm(prompt: str) -> str:
    _load_env()
    client = _make_openai_client()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def generate_free_loss_candidate(
    generation_prompt_path: str,
    *,
    operator_whitelist: Sequence[str],
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    del operator_whitelist
    base_prompt = _read_prompt(generation_prompt_path)
    prompt = base_prompt
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def crossover_free_loss(
    crossover_prompt_path: str,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(crossover_prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blob = {
            "index": idx,
            "name": parent.name,
            "intuition": parent.intuition,
            "pseudocode": parent.pseudocode,
            "hyperparams": parent.hyperparams,
            "operators_used": parent.operators_used,
            "code": parent.code,
            "theoretical_basis": getattr(parent, "theoretical_basis", ""),
            "metrics": {
                "hf_like_score": float(metrics.get("hf_like_score", float("inf")))
                if metrics
                else None,
                "validation_objective": float(metrics.get("validation_objective", float("inf")))
                if metrics
                else None,
                "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
                if metrics
                else None,
                "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            },
        }
        parent_blobs.append(blob)
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def mutate_free_loss(
    mutation_prompt_path: str,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(mutation_prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
            if metrics
            else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def e2_free_loss(
    e2_prompt_path: str,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """E2: consensus extraction over p parents, then synthesize a new child loss."""

    prompt = _read_prompt(e2_prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blob = {
            "index": idx,
            "name": parent.name,
            "intuition": parent.intuition,
            "pseudocode": parent.pseudocode,
            "hyperparams": parent.hyperparams,
            "operators_used": parent.operators_used,
            "code": parent.code,
            "theoretical_basis": getattr(parent, "theoretical_basis", ""),
            "metrics": {
                "hf_like_score": float(metrics.get("hf_like_score", float("inf")))
                if metrics
                else None,
                "validation_objective": float(metrics.get("validation_objective", float("inf")))
                if metrics
                else None,
                "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
                if metrics
                else None,
                "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            },
        }
        parent_blobs.append(blob)

    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob

    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def m2_tune_hparams(
    m2_prompt_path: str,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """M2: hyperparameter-only tuning; structurally identical to parent."""

    prompt = _read_prompt(m2_prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
            if metrics
            else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob

    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    tuned = parse_free_loss_from_text(json_str)

    parent_hp = dict(parent.hyperparams or {})
    tuned_hp = dict(tuned.hyperparams or {})

    # Enforce "hyperparams-only": keep structure and restrict hyperparam keys.
    if parent.code.strip():
        # If the parent has explicit code, only keep keys that already exist,
        # since new keys won't be used unless the code changes (which is forbidden in M2).
        tuned_hp = {k: tuned_hp.get(k, parent_hp.get(k)) for k in parent_hp.keys()}
    else:
        # Template-based losses may rely on compiler-known hyperparams; keep tuned as-is.
        tuned_hp = tuned_hp or parent_hp

    return FreeLossIR(
        name=tuned.name or f"{parent.name}_m2",
        intuition=tuned.intuition or parent.intuition,
        pseudocode=tuned.pseudocode or parent.pseudocode,
        hyperparams=tuned_hp,
        operators_used=list(parent.operators_used),
        implementation_hint=parent.implementation_hint,
        code=parent.code,
        theoretical_basis=tuned.theoretical_basis or getattr(parent, "theoretical_basis", ""),
    )


def m3_simplify_loss(
    m3_prompt_path: str,
    candidate: FreeLossIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """M3: simplify/stabilize a candidate loss, given a failure reason."""

    prompt = _read_prompt(m3_prompt_path)
    payload = {
        "candidate": {
            "name": candidate.name,
            "intuition": candidate.intuition,
            "pseudocode": candidate.pseudocode,
            "hyperparams": candidate.hyperparams,
            "operators_used": candidate.operators_used,
            "code": candidate.code,
            "theoretical_basis": getattr(candidate, "theoretical_basis", ""),
        },
        "failure_reason": dict(failure_reason),
    }
    prompt = prompt + "\n\nCANDIDATE_AND_FAILURE_JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob

    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    simplified = parse_free_loss_from_text(json_str)

    # Preserve the contract required by gates, even if the model drifts.
    simplified.implementation_hint = candidate.implementation_hint
    return simplified


def repair_free_loss(
    repair_prompt_path: str,
    failed_ir: FreeLossIR,
    failure_reason: Mapping[str, Any],
) -> FreeLossIR:
    prompt = _read_prompt(repair_prompt_path)
    payload = {
        "candidate": {
            "name": failed_ir.name,
            "intuition": failed_ir.intuition,
            "pseudocode": failed_ir.pseudocode,
            "hyperparams": failed_ir.hyperparams,
            "operators_used": failed_ir.operators_used,
            "code": failed_ir.code,
            "theoretical_basis": getattr(failed_ir, "theoretical_basis", ""),
        },
        "failure_reason": failure_reason,
    }
    prompt = prompt + "\n\nCANDIDATE_AND_FAILURE_JSON:\n" + json.dumps(payload, indent=2)
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def repair_expects_with_prompt(
    expects_repair_prompt_path: str,
    ir: FreeLossIR,
) -> FreeLossIR:
    """Use a lightweight LLM prompt to normalize implementation_hint.expects.

    This is only used when we already have an expects list, to coerce it into
    a clean list of short input names.
    """

    prompt = _read_prompt(expects_repair_prompt_path)
    payload = asdict(ir)
    prompt = prompt + "\n\nIR_JSON:\n" + json.dumps(payload, indent=2)
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def compile_free_loss_candidate(
    ir: FreeLossIR,
    *,
    operator_whitelist: Sequence[str],
) -> CompiledFreeLoss:
    return compile_free_loss(ir, operator_whitelist=operator_whitelist)
