from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Sequence

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
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


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
) -> FreeLossIR:
    del operator_whitelist
    base_prompt = _read_prompt(generation_prompt_path)
    raw = _call_llm(base_prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def crossover_free_loss(
    crossover_prompt_path: str,
    parents: Sequence[FreeLossIR],
) -> FreeLossIR:
    prompt = _read_prompt(crossover_prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        blob = {
            "index": idx,
            "name": parent.name,
            "intuition": parent.intuition,
            "pseudocode": parent.pseudocode,
            "hyperparams": parent.hyperparams,
            "operators_used": parent.operators_used,
            "code": parent.code,
        }
        parent_blobs.append(blob)
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2)
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def mutate_free_loss(
    mutation_prompt_path: str,
    parent: FreeLossIR,
) -> FreeLossIR:
    prompt = _read_prompt(mutation_prompt_path)
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2)
    raw = _call_llm(prompt)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def repair_free_loss(
    repair_prompt_path: str,
    failed_ir: FreeLossIR,
    failure_reason: str,
) -> FreeLossIR:
    prompt = _read_prompt(repair_prompt_path)
    payload = {
        "candidate": {
            "name": failed_ir.name,
            "intuition": failed_ir.intuition,
            "pseudocode": failed_ir.pseudocode,
            "hyperparams": failed_ir.hyperparams,
            "operators_used": failed_ir.operators_used,
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
