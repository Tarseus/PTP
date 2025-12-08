from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Set

from .ast import PTPProgramSpec
from .primitives import ANCHOR_PRIMITIVES, BUILD_PREF_PRIMITIVES, WEIGHT_PRIMITIVES


FORBIDDEN_FEATURE_NAMES: Set[str] = {
    "rank",
    "is_best",
    "cost_to_go",
    "advantage",
    "reward_to_go",
}


def _ensure_known_primitive(name: str, registry: Mapping[str, object], kind: str) -> None:
    if name not in registry:
        raise ValueError(f"Unknown {kind} primitive: {name}")


def _ensure_no_forbidden_features(params: Mapping[str, object]) -> None:
    """Static check to ensure DSL does not refer to supervision labels.

    We conservatively scan all string-valued fields and forbid known label-like
    names such as 'rank', 'is_best', 'cost_to_go', etc.
    """

    for value in params.values():
        if isinstance(value, str) and value in FORBIDDEN_FEATURE_NAMES:
            raise ValueError(
                f"PTP DSL may not reference forbidden feature '{value}'. "
                "Preference strategies must rely only on solution metadata "
                "and structural features, not supervision labels."
            )
        if isinstance(value, dict):
            _ensure_no_forbidden_features(value)
        elif isinstance(value, (list, tuple)):
            for element in value:
                if isinstance(element, dict):
                    _ensure_no_forbidden_features(element)


def _ensure_complexity_bounds(spec: PTPProgramSpec) -> None:
    """Sanity-check configuration fields that could induce O(N^2) behaviour.

    The primitives themselves are implemented to avoid unbounded O(N^2)
    enumeration of pairs, but we still cap key hyperparameters to safe ranges.
    """

    # Anchor primitives
    anchor_params = spec.anchors.params
    if spec.anchors.primitive == "best_of_k":
        k = int(anchor_params.get("k", 0))
        if k < 0 or k > 1024:
            raise ValueError("best_of_k.k must satisfy 0 <= k <= 1024.")

    if spec.anchors.primitive == "diverse_elites":
        num_anchors = int(anchor_params.get("num_anchors", 0))
        candidate_pool_size = int(anchor_params.get("candidate_pool_size", 64))
        if num_anchors < 0 or num_anchors > candidate_pool_size:
            raise ValueError(
                "diverse_elites.num_anchors must be in [0, candidate_pool_size]."
            )
        if candidate_pool_size > 256:
            raise ValueError(
                "diverse_elites.candidate_pool_size must be <= 256 to keep "
                "pairwise computations bounded."
            )

    # Preference primitives
    build_params = spec.build_preferences.params
    if spec.build_preferences.primitive == "best_anchored_pairs":
        max_pairs = int(build_params.get("max_pairs_per_anchor", 32))
        if max_pairs <= 0 or max_pairs > 256:
            raise ValueError(
                "best_anchored_pairs.max_pairs_per_anchor must be in (0, 256]."
            )

    if spec.build_preferences.primitive == "topk_vs_random":
        topk = int(build_params.get("topk", 8))
        pairs_per_topk = int(build_params.get("pairs_per_topk", 16))
        if topk < 0 or topk > 1024:
            raise ValueError("topk_vs_random.topk must satisfy 0 <= topk <= 1024.")
        if pairs_per_topk <= 0 or pairs_per_topk > 256:
            raise ValueError(
                "topk_vs_random.pairs_per_topk must be in (0, 256]."
            )

    if spec.build_preferences.primitive == "hardness_bounded_pairs":
        max_pairs = int(build_params.get("max_pairs", 1024))
        if max_pairs <= 0 or max_pairs > 8192:
            raise ValueError(
                "hardness_bounded_pairs.max_pairs must be in (0, 8192]."
            )

    if spec.build_preferences.primitive == "diversity_contrast_pairs":
        num_pairs = int(build_params.get("num_pairs", 256))
        candidate_pool_size = int(build_params.get("candidate_pool_size", 128))
        if num_pairs <= 0 or num_pairs > 4096:
            raise ValueError(
                "diversity_contrast_pairs.num_pairs must be in (0, 4096]."
            )
        if candidate_pool_size > 256:
            raise ValueError(
                "diversity_contrast_pairs.candidate_pool_size must be <= 256."
            )


def validate_ptp_program_spec(spec: PTPProgramSpec) -> None:
    """Run static checks on a PTP program specification.

    This ensures:
        - all primitives are known
        - no forbidden supervision labels are referenced
        - configuration fields that could lead to O(N^2) behaviour are bounded
    """

    _ensure_known_primitive(spec.anchors.primitive, ANCHOR_PRIMITIVES, "anchor")
    _ensure_known_primitive(
        spec.build_preferences.primitive,
        BUILD_PREF_PRIMITIVES,
        "build_preferences",
    )
    _ensure_known_primitive(spec.weight.primitive, WEIGHT_PRIMITIVES, "weight")

    _ensure_no_forbidden_features(spec.anchors.params)
    _ensure_no_forbidden_features(spec.build_preferences.params)
    _ensure_no_forbidden_features(spec.weight.params)
    _ensure_no_forbidden_features(spec.schedule)

    _ensure_complexity_bounds(spec)

