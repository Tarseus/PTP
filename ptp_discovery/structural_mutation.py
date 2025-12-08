from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List, Optional


def _load_dsl(source: str) -> Dict[str, Any]:
    return json.loads(source)


def _dump_dsl(dsl_obj: Dict[str, Any]) -> str:
    return json.dumps(dsl_obj, indent=2, sort_keys=True)


def mutate_weight_thresholds(source: str) -> Optional[str]:
    """Structured mutation: adjust thresholds inside the weight specification."""

    try:
        obj = _load_dsl(source)
    except json.JSONDecodeError:
        return None

    weight = obj.get("weight", {})
    params = weight.get("params", weight) if isinstance(weight, dict) else {}

    mutated = False

    # Piecewise linear: jitter knot y-values slightly.
    knots = params.get("knots")
    if isinstance(knots, list) and knots:
        for knot in knots:
            if "y" in knot:
                jitter = random.uniform(-0.2, 0.2)
                knot["y"] = float(knot["y"]) + jitter
                mutated = True

    # Soft threshold: perturb threshold.
    if "threshold" in params:
        jitter = random.uniform(-0.1, 0.1)
        params["threshold"] = float(params["threshold"]) + jitter
        mutated = True

    if not mutated:
        return None

    if "params" in weight:
        weight["params"] = params
    else:
        obj["weight"] = params
    return _dump_dsl(obj)


def swap_build_preferences_primitive(source: str) -> Optional[str]:
    """Structured mutation: swap the build_preferences primitive."""

    try:
        obj = _load_dsl(source)
    except json.JSONDecodeError:
        return None

    build = obj.get("build_preferences")
    if not isinstance(build, dict):
        return None

    current = build.get("primitive")
    candidates = [
        "best_anchored_pairs",
        "topk_vs_random",
        "hardness_bounded_pairs",
        "diversity_contrast_pairs",
    ]
    if current not in candidates:
        return None

    alternatives = [name for name in candidates if name != current]
    if not alternatives:
        return None

    build["primitive"] = random.choice(alternatives)
    obj["build_preferences"] = build
    return _dump_dsl(obj)


def mutate_schedule_boundaries(source: str) -> Optional[str]:
    """Structured mutation: slightly move the fractional boundaries of stages."""

    try:
        obj = _load_dsl(source)
    except json.JSONDecodeError:
        return None

    schedule = obj.get("schedule")
    if not isinstance(schedule, dict):
        return None

    stages = schedule.get("stages")
    if not isinstance(stages, list) or not stages:
        return None

    mutated = False
    for stage in stages:
        if "start_frac" in stage and "end_frac" in stage:
            width = float(stage["end_frac"] - stage["start_frac"])
            if width <= 0:
                continue
            shift = random.uniform(-0.05 * width, 0.05 * width)
            stage["start_frac"] = max(0.0, float(stage["start_frac"]) + shift)
            stage["end_frac"] = min(1.0, float(stage["end_frac"]) + shift)
            mutated = True

    if not mutated:
        return None

    schedule["stages"] = stages
    obj["schedule"] = schedule
    return _dump_dsl(obj)


def crossover_modules(source_a: str, source_b: str) -> Optional[str]:
    """Module-level crossover: combine anchors from A and weighting from B."""

    try:
        obj_a = _load_dsl(source_a)
        obj_b = _load_dsl(source_b)
    except json.JSONDecodeError:
        return None

    anchors = obj_a.get("anchors")
    build_preferences = obj_a.get("build_preferences")
    weight = obj_b.get("weight")

    if not isinstance(anchors, dict) or not isinstance(build_preferences, dict) or not isinstance(weight, dict):
        return None

    child = {
        "anchors": anchors,
        "build_preferences": build_preferences,
        "weight": weight,
    }

    # Prefer schedule from A if present, otherwise from B.
    schedule = obj_a.get("schedule", obj_b.get("schedule"))
    if isinstance(schedule, dict):
        child["schedule"] = schedule

    return _dump_dsl(child)

