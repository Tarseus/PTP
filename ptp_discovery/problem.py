from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from fitness.ptp_high_fidelity import (
    HighFidelityConfig,
    evaluate_ptp_dsl_high_fidelity,
)
from ptp_dsl.compiler import emit_ptp_program_python, parse_ptp_dsl


@dataclass
class PTPDiscoveryCandidate:
    """A single candidate PTP program expressed in the DSL."""

    dsl_source: str
    origin: str = "unknown"  # e.g. "llm", "mutation", "crossover"
    parent_ids: Optional[list[str]] = None


@dataclass
class PTPDiscoveryResult:
    """Evaluation record for a candidate."""

    candidate_id: str
    hf_score: float
    validation_objective: float
    generalization_penalty: float
    generalization_objectives: Dict[int, float]
    hf_raw: Dict[str, Any]


class PTPDiscoveryProblem:
    """Problem-level wrapper used by search to evaluate PTP candidates.

    This class connects the PTP DSL to the high-fidelity PO4COPs / POMO
    training pipeline and handles logging of artifacts required for
    reproducibility:
        - DSL source
        - compiled Python module
        - high-fidelity metrics and HF_score
    """

    def __init__(self, hf_config: HighFidelityConfig, log_dir: str) -> None:
        self.hf_config = hf_config
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._counter = 0

    def _next_candidate_id(self) -> str:
        self._counter += 1
        return f"ptp_{self._counter:05d}"

    def evaluate(self, candidate: PTPDiscoveryCandidate) -> PTPDiscoveryResult:
        candidate_id = self._next_candidate_id()
        hf_raw = evaluate_ptp_dsl_high_fidelity(candidate.dsl_source, self.hf_config)

        result = PTPDiscoveryResult(
            candidate_id=candidate_id,
            hf_score=float(hf_raw["hf_score"]),
            validation_objective=float(hf_raw["validation_objective"]),
            generalization_penalty=float(hf_raw["generalization_penalty"]),
            generalization_objectives={
                int(k): float(v)
                for k, v in hf_raw.get("generalization_objectives", {}).items()
            },
            hf_raw=hf_raw,
        )

        self._log_candidate(candidate_id, candidate, result)
        return result

    def _log_candidate(
        self,
        candidate_id: str,
        candidate: PTPDiscoveryCandidate,
        result: PTPDiscoveryResult,
    ) -> None:
        """Persist DSL, compiled Python, and evaluation metrics."""

        candidate_dir = os.path.join(self.log_dir, candidate_id)
        os.makedirs(candidate_dir, exist_ok=True)

        # Raw DSL source.
        dsl_path = os.path.join(candidate_dir, "program.dsl.json")
        with open(dsl_path, "w", encoding="utf-8") as f:
            f.write(candidate.dsl_source)

        # Compiled Python wrapper.
        spec = parse_ptp_dsl(candidate.dsl_source)
        compiled_py = emit_ptp_program_python(spec)
        compiled_path = os.path.join(candidate_dir, "program_compiled.py")
        with open(compiled_path, "w", encoding="utf-8") as f:
            f.write(compiled_py)

        # Metrics and meta-data.
        metrics = {
            "candidate": asdict(candidate),
            "result": asdict(result),
            "hf_raw": result.hf_raw,
        }
        metrics_path = os.path.join(candidate_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

