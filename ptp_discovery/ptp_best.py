from __future__ import annotations

"""Canonical representation of the current best-known PTP program.

This module is intended to be overwritten or updated by the search
pipeline once a new best candidate is discovered. For now, it hosts a
reasonable hand-written baseline in the DSL.
"""

from ptp_dsl import parse_ptp_dsl, compile_ptp_program


PTP_BEST_DSL = """
{
  "anchors": {
    "primitive": "best_of_k",
    "k": 8
  },
  "build_preferences": {
    "primitive": "topk_vs_random",
    "topk": 8,
    "pairs_per_topk": 16
  },
  "weight": {
    "primitive": "logistic",
    "w_delta_obj": 1.0,
    "w_delta_struct": 0.5,
    "w_size": 0.0,
    "bias": 0.0,
    "stage_multipliers": {
      "warmup": 0.5,
      "main": 1.0,
      "finetune": 0.5
    }
  },
  "schedule": {
    "stages": [
      {"name": "warmup", "start_frac": 0.0, "end_frac": 0.3},
      {"name": "main", "start_frac": 0.3, "end_frac": 0.8},
      {"name": "finetune", "start_frac": 0.8, "end_frac": 1.0}
    ]
  }
}
""".strip()

_spec = parse_ptp_dsl(PTP_BEST_DSL)
_compiled = compile_ptp_program(_spec)

select_anchors = _compiled.select_anchors
build_preferences = _compiled.build_preferences
weight_preference = _compiled.weight_preference

