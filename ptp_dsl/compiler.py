from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .ast import AnchorSpec, BuildPreferencesSpec, PTPProgramSpec, WeightSpec
from .primitives import (
    SolutionId,
    get_anchor_primitive,
    get_build_preferences_primitive,
    get_weight_primitive,
)
from .validators import validate_ptp_program_spec


SelectAnchorsFn = Callable[[Mapping[str, Any], Mapping[str, Any], str], List[SolutionId]]
BuildPreferencesFn = Callable[
    [Sequence[SolutionId], Mapping[str, Any], str],
    Sequence[Tuple[SolutionId, SolutionId]],
]
WeightPreferenceFn = Callable[[float, float, float, str], float]


@dataclass
class CompiledPTPProgram:
    """Compiled PTP program with executable callables."""

    spec: PTPProgramSpec
    select_anchors: SelectAnchorsFn
    build_preferences: BuildPreferencesFn
    weight_preference: WeightPreferenceFn


def parse_ptp_dsl(source: str) -> PTPProgramSpec:
    """Parse a PTP DSL source string into a structured specification.

    The current implementation expects a JSON object of the form:

    {
      "anchors": {"primitive": "best_of_k", "params": {...}},
      "build_preferences": {"primitive": "topk_vs_random", "params": {...}},
      "weight": {"primitive": "logistic", "params": {...}},
      "schedule": {...}   # optional
    }
    """

    try:
        raw = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse PTP DSL JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Top-level PTP DSL must be a JSON object.")

    anchors_raw = raw.get("anchors")
    build_raw = raw.get("build_preferences")
    weight_raw = raw.get("weight")
    schedule = raw.get("schedule", {}) or {}

    if not isinstance(anchors_raw, dict) or "primitive" not in anchors_raw:
        raise ValueError("anchors must be an object with a 'primitive' field.")
    if not isinstance(build_raw, dict) or "primitive" not in build_raw:
        raise ValueError(
            "build_preferences must be an object with a 'primitive' field."
        )
    if not isinstance(weight_raw, dict) or "primitive" not in weight_raw:
        raise ValueError("weight must be an object with a 'primitive' field.")

    # Support both flattened parameters:
    #   {"primitive": "best_of_k", "k": 8}
    # and nested "params" objects:
    #   {"primitive": "best_of_k", "params": {"k": 8}}
    def _extract_params(obj: Mapping[str, Any]) -> Dict[str, Any]:
        params_field = obj.get("params")
        if isinstance(params_field, dict):
            # Prefer explicit nested params block.
            return dict(params_field)
        # Fallback to treating all non-primitive, non-params keys as params.
        return {k: v for k, v in obj.items() if k not in ("primitive", "params")}

    anchors = AnchorSpec(
        primitive=str(anchors_raw["primitive"]),
        params=_extract_params(anchors_raw),
    )
    build_preferences = BuildPreferencesSpec(
        primitive=str(build_raw["primitive"]),
        params=_extract_params(build_raw),
    )
    weight = WeightSpec(
        primitive=str(weight_raw["primitive"]),
        params=_extract_params(weight_raw),
    )

    spec = PTPProgramSpec(
        anchors=anchors,
        build_preferences=build_preferences,
        weight=weight,
        schedule=schedule,
        source=source,
    )

    validate_ptp_program_spec(spec)
    return spec


def compile_ptp_program(spec: PTPProgramSpec) -> CompiledPTPProgram:
    """Compile a PTPProgramSpec into executable Python callables."""

    validate_ptp_program_spec(spec)

    anchor_primitive_fn = get_anchor_primitive(spec.anchors.primitive)
    build_pref_primitive_fn = get_build_preferences_primitive(
        spec.build_preferences.primitive
    )
    weight_primitive_fn = get_weight_primitive(spec.weight.primitive)

    def select_anchors(
        instance_meta: Mapping[str, Any],
        pool_meta: Mapping[str, Any],
        stage: str,
    ) -> List[SolutionId]:
        return anchor_primitive_fn(
            instance_meta=instance_meta,
            pool_meta=pool_meta,
            stage=stage,
            **spec.anchors.params,
        )

    def build_preferences(
        anchor_ids: Sequence[SolutionId],
        pool_meta: Mapping[str, Any],
        stage: str,
    ) -> List[Tuple[SolutionId, SolutionId]]:
        pairs = build_pref_primitive_fn(
            anchor_ids=anchor_ids,
            pool_meta=pool_meta,
            stage=stage,
            **spec.build_preferences.params,
        )
        return list(pairs)

    def weight_preference(
        delta_obj: float,
        delta_struct: float,
        size: float,
        stage: str,
    ) -> float:
        return float(
            weight_primitive_fn(
                delta_obj=delta_obj,
                delta_struct=delta_struct,
                size=size,
                stage=stage,
                **spec.weight.params,
            )
        )

    return CompiledPTPProgram(
        spec=spec,
        select_anchors=select_anchors,
        build_preferences=build_preferences,
        weight_preference=weight_preference,
    )


def emit_ptp_program_python(spec: PTPProgramSpec) -> str:
    """Emit a self-contained Python snippet that recreates the compiled
    PTP program from its DSL source.

    This is primarily intended for logging and reproducibility rather than
    performance-critical execution.
    """

    source = spec.source or ""
    # Escape triple quotes conservatively.
    safe_source = source.replace('"""', r"\"\"\"")

    return f'''"""
Auto-generated PTP candidate module.

This file was produced by ptp_dsl.compiler.emit_ptp_program_python and
captures the DSL source plus the three canonical entry points:
    - select_anchors(instance_meta, pool_meta, stage)
    - build_preferences(anchor_ids, pool_meta, stage)
    - weight_preference(delta_obj, delta_struct, size, stage)
"""

from ptp_dsl import parse_ptp_dsl, compile_ptp_program

PTP_DSL_SOURCE = """{safe_source}"""

_spec = parse_ptp_dsl(PTP_DSL_SOURCE)
_compiled = compile_ptp_program(_spec)

select_anchors = _compiled.select_anchors
build_preferences = _compiled.build_preferences
weight_preference = _compiled.weight_preference
'''
