from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnchorSpec:
    """Specification for how to select anchors from a solution pool.

    The `primitive` field must be one of the supported anchor primitives, such as:
        - "best_of_k"
        - "elite_percentile"
        - "diverse_elites"
        - "size_conditional_best"

    The remaining configuration is stored in `params` and interpreted by the
    corresponding primitive implementation.
    """

    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildPreferencesSpec:
    """Specification for how to build preference pairs or lists.

    The `primitive` field must be one of:
        - "best_anchored_pairs"
        - "topk_vs_random"
        - "hardness_bounded_pairs"
        - "diversity_contrast_pairs"
    """

    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightSpec:
    """Specification for how to weight each preference example.

    This operates on the features:
        - delta_obj:    scalar objective difference
        - delta_struct: scalar structural difference
        - size:         instance size or solution size
        - stage:        training stage identifier

    Supported primitive types (extensible):
        - "piecewise_linear"
        - "soft_threshold"
        - "logistic"
        - "composite" (sum/product of several sub-weights)

    The exact semantics are implemented in `primitives.py`.
    """

    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PTPProgramSpec:
    """Full preference teaching program specification.

    A candidate PTP program consists of:
        - an anchor selection strategy
        - a preference construction strategy
        - a preference weighting strategy
        - an optional stage schedule describing training phases

    The *source* field stores the raw DSL text (JSON string in the current
    implementation) for logging and reproducibility.
    """

    anchors: AnchorSpec
    build_preferences: BuildPreferencesSpec
    weight: WeightSpec
    schedule: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

