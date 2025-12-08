from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SolutionId = Any


@dataclass
class SolutionMeta:
    """Light-weight metadata for a single solution in the pool.

    Required fields:
        - solution_id: identifier used to join back with full solutions
        - objective:   scalar objective value (lower is better)
        - size:        scalar size (e.g., number of nodes)

    Optional fields:
        - struct_repr: arbitrary structural representation used to compute
                       structural deltas (e.g., edge sets, embeddings, etc.)
        - extra:       any additional meta-features (distribution stats, etc.)
    """

    solution_id: SolutionId
    objective: float
    size: int
    struct_repr: Any = None
    extra: Dict[str, Any] = None


# A pool is represented as a dict with at least the "solutions" key.
SolutionPoolMeta = Mapping[str, Any]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sorted_solutions_by_objective(pool_meta: SolutionPoolMeta) -> List[SolutionMeta]:
    solutions: Sequence[Mapping[str, Any]] = pool_meta.get("solutions", [])
    return sorted(
        (
            SolutionMeta(
                solution_id=sol["solution_id"],
                objective=float(sol["objective"]),
                size=int(sol.get("size", sol.get("problem_size", 0))),
                struct_repr=sol.get("struct_repr"),
                extra=sol.get("extra"),
            )
            for sol in solutions
        ),
        key=lambda s: s.objective,
    )


def _sample_without_replacement(items: Sequence[Any], k: int) -> List[Any]:
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    return random.sample(list(items), k)


def _estimate_struct_delta(a: SolutionMeta, b: SolutionMeta) -> float:
    """Estimate structural difference between two solutions.

    The implementation is intentionally conservative and relies only on
    structural metadata provided in `struct_repr`. For example, if `struct_repr`
    is a set of edges, we compute a Jaccard distance; if it is a vector-like
    sequence, we fall back to a normalized Hamming distance.
    """

    sa, sb = a.struct_repr, b.struct_repr

    if sa is None or sb is None:
        return 0.0

    if isinstance(sa, set) and isinstance(sb, set):
        if not sa and not sb:
            return 0.0
        intersection = len(sa.intersection(sb))
        union = len(sa.union(sb))
        return 1.0 - intersection / max(union, 1)

    if isinstance(sa, (list, tuple)) and isinstance(sb, (list, tuple)):
        length = min(len(sa), len(sb))
        if length == 0:
            return 0.0
        mismatch = sum(1 for i in range(length) if sa[i] != sb[i])
        mismatch += abs(len(sa) - len(sb))
        return mismatch / float(max(len(sa), len(sb)))

    # Fallback: treat unequal representations as maximally different.
    return 1.0


# ---------------------------------------------------------------------------
# Anchor selection primitives
# ---------------------------------------------------------------------------

def select_anchors_best_of_k(
    instance_meta: Mapping[str, Any],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    k: int,
) -> List[SolutionId]:
    """Select the globally best-k solutions by objective."""

    del instance_meta, stage  # unused in this primitive
    sorted_solutions = _sorted_solutions_by_objective(pool_meta)
    selected = sorted_solutions[: max(k, 0)]
    return [s.solution_id for s in selected]


def select_anchors_elite_percentile(
    instance_meta: Mapping[str, Any],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    percentile: float,
    min_anchors: int = 1,
) -> List[SolutionId]:
    """Select all solutions within an elite percentile by objective."""

    del instance_meta, stage
    percentile = max(0.0, min(1.0, float(percentile)))
    sorted_solutions = _sorted_solutions_by_objective(pool_meta)
    if not sorted_solutions:
        return []

    cutoff_index = max(int(len(sorted_solutions) * percentile), min_anchors)
    cutoff_index = min(cutoff_index, len(sorted_solutions))
    selected = sorted_solutions[:cutoff_index]
    return [s.solution_id for s in selected]


def select_anchors_diverse_elites(
    instance_meta: Mapping[str, Any],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    num_anchors: int,
    candidate_pool_size: int = 64,
    random_seed: int = 0,
) -> List[SolutionId]:
    """Select a small set of elite but structurally diverse anchors.

    To respect the O(N^2) constraint, we only perform diversity-aware
    selection on a capped subset of the top candidates.
    """

    del instance_meta, stage
    random.seed(random_seed)

    sorted_solutions = _sorted_solutions_by_objective(pool_meta)
    if not sorted_solutions:
        return []

    candidate_pool = sorted_solutions[: max(candidate_pool_size, 1)]
    if len(candidate_pool) <= num_anchors:
        return [s.solution_id for s in candidate_pool]

    anchors: List[SolutionMeta] = []
    # Start from the very best solution.
    anchors.append(candidate_pool[0])

    # Greedy farthest-first traversal on the capped candidate pool.
    while len(anchors) < num_anchors:
        best_candidate = None
        best_distance = -1.0
        for candidate in candidate_pool:
            if candidate in anchors:
                continue
            # Distance to the closest existing anchor.
            min_delta = min(
                _estimate_struct_delta(candidate, existing) for existing in anchors
            )
            if min_delta > best_distance:
                best_distance = min_delta
                best_candidate = candidate

        if best_candidate is None:
            break
        anchors.append(best_candidate)

    return [s.solution_id for s in anchors]


def select_anchors_size_conditional_best(
    instance_meta: Mapping[str, Any],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    size_bins: Sequence[Mapping[str, Any]],
) -> List[SolutionId]:
    """Select best solutions conditioned on solution or instance size.

    `size_bins` is a list of dicts with:
        - "max_size": upper bound (inclusive) for the bin
        - "k":        number of anchors to select within that bin
    """

    del stage
    sorted_solutions = _sorted_solutions_by_objective(pool_meta)
    if not sorted_solutions:
        return []

    # Determine the size attribute (instance-level or solution-level).
    instance_size = int(instance_meta.get("size", 0))

    anchors: List[SolutionId] = []
    remaining = list(sorted_solutions)

    for bin_conf in size_bins:
        max_size = int(bin_conf.get("max_size", 0))
        k = int(bin_conf.get("k", 0))
        if k <= 0:
            continue

        # Bin condition: either by instance size or solution size.
        if instance_size and instance_size > max_size:
            continue

        bin_candidates = [s for s in remaining if s.size <= max_size]
        take = bin_candidates[:k]
        anchors.extend(s.solution_id for s in take)

        remaining = [s for s in remaining if s not in take]
        if not remaining:
            break

    return anchors


# ---------------------------------------------------------------------------
# Preference construction primitives
# ---------------------------------------------------------------------------

def build_preferences_best_anchored_pairs(
    anchor_ids: Sequence[SolutionId],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    max_pairs_per_anchor: int = 32,
) -> List[Tuple[SolutionId, SolutionId]]:
    """For each anchor, pair it with a bounded number of contrasting solutions.

    Pairs are directional (better, worse) and expressed as (winner, loser).
    """

    del stage
    solutions = _sorted_solutions_by_objective(pool_meta)
    by_id = {s.solution_id: s for s in solutions}
    anchors = [by_id[a] for a in anchor_ids if a in by_id]

    pairs: List[Tuple[SolutionId, SolutionId]] = []
    if not anchors or not solutions:
        return pairs

    for anchor in anchors:
        worse_candidates = [s for s in solutions if s.objective > anchor.objective]
        if not worse_candidates:
            continue
        chosen_worse = _sample_without_replacement(
            worse_candidates, max_pairs_per_anchor
        )
        for loser in chosen_worse:
            pairs.append((anchor.solution_id, loser.solution_id))

    return pairs


def build_preferences_topk_vs_random(
    anchor_ids: Sequence[SolutionId],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    topk: int = 8,
    pairs_per_topk: int = 16,
) -> List[Tuple[SolutionId, SolutionId]]:
    """Pairs top-k solutions against random pool members.

    This encourages global ranking without enumerating all O(N^2) pairs.
    """

    del stage, anchor_ids
    solutions = _sorted_solutions_by_objective(pool_meta)
    if not solutions:
        return []

    top_solutions = solutions[: max(topk, 0)]
    remaining = solutions[len(top_solutions) :]

    if not remaining:
        return []

    pairs: List[Tuple[SolutionId, SolutionId]] = []
    for top_solution in top_solutions:
        opponents = _sample_without_replacement(remaining, pairs_per_topk)
        for opponent in opponents:
            better, worse = (
                (top_solution.solution_id, opponent.solution_id)
                if top_solution.objective <= opponent.objective
                else (opponent.solution_id, top_solution.solution_id)
            )
            pairs.append((better, worse))

    return pairs


def build_preferences_hardness_bounded_pairs(
    anchor_ids: Sequence[SolutionId],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    min_delta_obj: float = 0.0,
    max_delta_obj: float = 1.0,
    max_pairs: int = 1024,
) -> List[Tuple[SolutionId, SolutionId]]:
    """Construct pairs whose objective gap lies in a target interval."""

    del anchor_ids, stage
    solutions = _sorted_solutions_by_objective(pool_meta)
    if not solutions:
        return []

    pairs: List[Tuple[SolutionId, SolutionId]] = []

    # Efficient sweep on sorted objectives to avoid O(N^2) enumeration.
    left = 0
    n = len(solutions)
    for right in range(n):
        while (
            left < right
            and solutions[right].objective - solutions[left].objective > max_delta_obj
        ):
            left += 1
        cursor = left
        while (
            cursor < right
            and solutions[right].objective - solutions[cursor].objective
            >= min_delta_obj
        ):
            better = solutions[cursor]
            worse = solutions[right]
            pairs.append((better.solution_id, worse.solution_id))
            cursor += 1
            if len(pairs) >= max_pairs:
                return pairs

    return pairs


def build_preferences_diversity_contrast_pairs(
    anchor_ids: Sequence[SolutionId],
    pool_meta: SolutionPoolMeta,
    stage: str,
    *,
    num_pairs: int = 256,
    min_delta_struct: float = 0.3,
    candidate_pool_size: int = 128,
) -> List[Tuple[SolutionId, SolutionId]]:
    """Build pairs that are structurally contrasting but not necessarily far
    apart in objective space.
    """

    del stage
    solutions = _sorted_solutions_by_objective(pool_meta)
    if not solutions:
        return []

    by_id = {s.solution_id: s for s in solutions}
    candidate_pool = solutions[: max(candidate_pool_size, 1)]

    anchors: List[SolutionMeta] = [
        by_id[a] for a in anchor_ids if a in by_id and by_id[a] in candidate_pool
    ]
    if not anchors:
        anchors = candidate_pool

    pairs: List[Tuple[SolutionId, SolutionId]] = []
    for anchor in anchors:
        for candidate in candidate_pool:
            if anchor is candidate:
                continue
            delta_struct = _estimate_struct_delta(anchor, candidate)
            if delta_struct < min_delta_struct:
                continue
            better, worse = (
                (anchor.solution_id, candidate.solution_id)
                if anchor.objective <= candidate.objective
                else (candidate.solution_id, anchor.solution_id)
            )
            pairs.append((better, worse))
            if len(pairs) >= num_pairs:
                return pairs

    return pairs


# ---------------------------------------------------------------------------
# Weighting primitives
# ---------------------------------------------------------------------------

def weight_piecewise_linear(
    delta_obj: float,
    delta_struct: float,
    size: float,
    stage: str,
    *,
    knots: Sequence[Mapping[str, float]],
    feature: str = "delta_obj",
) -> float:
    """Piecewise-linear weighting on a single feature."""

    if feature == "delta_obj":
        x = float(delta_obj)
    elif feature == "delta_struct":
        x = float(delta_struct)
    elif feature == "size":
        x = float(size)
    elif feature == "stage_index":
        # Allow stage-aware weighting through an integer-coded stage.
        x = float(hash(stage) % 10)
    else:
        raise ValueError(f"Unsupported feature for piecewise_linear: {feature}")

    if not knots:
        return 1.0

    sorted_knots = sorted(knots, key=lambda k: float(k["x"]))
    if x <= sorted_knots[0]["x"]:
        return float(sorted_knots[0]["y"])
    if x >= sorted_knots[-1]["x"]:
        return float(sorted_knots[-1]["y"])

    for left, right in zip(sorted_knots[:-1], sorted_knots[1:]):
        if left["x"] <= x <= right["x"]:
            t = (x - left["x"]) / max(right["x"] - left["x"], 1e-8)
            return float(left["y"] + t * (right["y"] - left["y"]))

    return 1.0


def weight_soft_threshold(
    delta_obj: float,
    delta_struct: float,
    size: float,
    stage: str,
    *,
    feature: str = "delta_obj",
    threshold: float = 0.0,
    sharpness: float = 10.0,
) -> float:
    """Soft-threshold weight that smoothly turns on past a threshold."""

    if feature == "delta_obj":
        x = float(delta_obj)
    elif feature == "delta_struct":
        x = float(delta_struct)
    elif feature == "size":
        x = float(size)
    else:
        raise ValueError(f"Unsupported feature for soft_threshold: {feature}")

    return 1.0 / (1.0 + math.exp(-sharpness * (x - threshold)))


def weight_logistic(
    delta_obj: float,
    delta_struct: float,
    size: float,
    stage: str,
    *,
    w_delta_obj: float = 0.0,
    w_delta_struct: float = 0.0,
    w_size: float = 0.0,
    bias: float = 0.0,
    stage_multipliers: Mapping[str, float] | None = None,
) -> float:
    """Logistic weighting on a linear combination of features."""

    z = (
        float(w_delta_obj) * float(delta_obj)
        + float(w_delta_struct) * float(delta_struct)
        + float(w_size) * float(size)
        + float(bias)
    )
    base = 1.0 / (1.0 + math.exp(-z))

    if stage_multipliers and stage in stage_multipliers:
        return base * float(stage_multipliers[stage])
    return base


def weight_composite(
    delta_obj: float,
    delta_struct: float,
    size: float,
    stage: str,
    *,
    components: Sequence[Mapping[str, Any]],
    combine_op: str = "product",
) -> float:
    """Combine several sub-weights via product or sum."""

    if not components:
        return 1.0

    weights: List[float] = []
    for component in components:
        primitive = component.get("primitive")
        # Support both flattened sub-params and nested "params" objects.
        if isinstance(component.get("params"), dict):
            params = dict(component["params"])
        else:
            params = {k: v for k, v in component.items() if k not in ("primitive", "params")}
        weight_fn = get_weight_primitive(primitive)
        weights.append(
            weight_fn(
                delta_obj=delta_obj,
                delta_struct=delta_struct,
                size=size,
                stage=stage,
                **params,
            )
        )

    if combine_op == "sum":
        return float(sum(weights))
    if combine_op == "product":
        prod = 1.0
        for value in weights:
            prod *= float(value)
        return prod

    raise ValueError(f"Unsupported combine_op for composite weight: {combine_op}")


# ---------------------------------------------------------------------------
# Primitive registry
# ---------------------------------------------------------------------------

ANCHOR_PRIMITIVES: Dict[str, Callable[..., List[SolutionId]]] = {
    "best_of_k": select_anchors_best_of_k,
    "elite_percentile": select_anchors_elite_percentile,
    "diverse_elites": select_anchors_diverse_elites,
    "size_conditional_best": select_anchors_size_conditional_best,
}

BUILD_PREF_PRIMITIVES: Dict[
    str, Callable[..., List[Tuple[SolutionId, SolutionId]]]
] = {
    "best_anchored_pairs": build_preferences_best_anchored_pairs,
    "topk_vs_random": build_preferences_topk_vs_random,
    "hardness_bounded_pairs": build_preferences_hardness_bounded_pairs,
    "diversity_contrast_pairs": build_preferences_diversity_contrast_pairs,
}

WEIGHT_PRIMITIVES: Dict[str, Callable[..., float]] = {
    "piecewise_linear": weight_piecewise_linear,
    "soft_threshold": weight_soft_threshold,
    "logistic": weight_logistic,
    "composite": weight_composite,
}


def get_anchor_primitive(name: str) -> Callable[..., List[SolutionId]]:
    if name not in ANCHOR_PRIMITIVES:
        raise KeyError(f"Unknown anchor primitive: {name}")
    return ANCHOR_PRIMITIVES[name]


def get_build_preferences_primitive(
    name: str,
) -> Callable[..., List[Tuple[SolutionId, SolutionId]]]:
    if name not in BUILD_PREF_PRIMITIVES:
        raise KeyError(f"Unknown build_preferences primitive: {name}")
    return BUILD_PREF_PRIMITIVES[name]


def get_weight_primitive(name: str) -> Callable[..., float]:
    if name not in WEIGHT_PRIMITIVES:
        raise KeyError(f"Unknown weight primitive: {name}")
    return WEIGHT_PRIMITIVES[name]
