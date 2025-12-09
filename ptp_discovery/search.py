from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from fitness.ptp_high_fidelity import HighFidelityConfig, evaluate_ptp_dsl_high_fidelity
from .problem import PTPDiscoveryCandidate, PTPDiscoveryResult, PTPDiscoveryProblem
from .structural_mutation import (
    swap_build_preferences_primitive,
    crossover_modules,
)


@dataclass
class EliteRecord:
    candidate: PTPDiscoveryCandidate
    result: PTPDiscoveryResult


class PTPDiscoverySearch:
    """Minimal search loop for PTP strategy discovery.

    This class is intentionally lightweight and framework-agnostic so that
    it can be plugged into an external HeuristicFinder orchestration layer.
    """

    def __init__(
        self,
        hf_config: HighFidelityConfig,
        log_dir: str,
        population_size: int = 8,
        elite_size: int = 4,
    ) -> None:
        self.problem = PTPDiscoveryProblem(hf_config=hf_config, log_dir=log_dir)
        self.population_size = population_size
        self.elite_size = elite_size
        self.elites: List[EliteRecord] = []

    def _update_elites(self, candidate: PTPDiscoveryCandidate, result: PTPDiscoveryResult) -> None:
        """Maintain a max-heap of elites based on HF_score (lower is better)."""

        record = EliteRecord(candidate=candidate, result=result)
        self.elites.append(record)
        # Sort by hf_score ascending (lower is better).
        self.elites.sort(key=lambda r: r.result.hf_score)
        if len(self.elites) > self.elite_size:
            self.elites = self.elites[: self.elite_size]

    def evaluate_generation(
        self, candidates: Sequence[PTPDiscoveryCandidate]
    ) -> List[EliteRecord]:
        """Evaluate a batch of candidates and update the elite pool."""

        for candidate in candidates:
            result = self.problem.evaluate(candidate)
            self._update_elites(candidate, result)
        return list(self.elites)

    def propose_mutations(self) -> List[PTPDiscoveryCandidate]:
        """Generate new candidates by structurally mutating current elites.

        This implements the requested structured mutation operators, without
        relying on arbitrary string perturbations.
        """

        if not self.elites:
            return []

        new_candidates: List[PTPDiscoveryCandidate] = []

        for elite_rec in self.elites:
            base_source = elite_rec.candidate.dsl_source

            mutated_sources = [
                mutate_weight_thresholds(base_source),
                swap_build_preferences_primitive(base_source),
            ]

            for mutated in mutated_sources:
                if mutated is None:
                    continue
                new_candidates.append(
                    PTPDiscoveryCandidate(
                        dsl_source=mutated,
                        origin="mutation",
                        parent_ids=[elite_rec.result.candidate_id],
                    )
                )

        # Module-level crossover between pairs of elites.
        for i in range(len(self.elites)):
            for j in range(i + 1, len(self.elites)):
                parent_a = self.elites[i]
                parent_b = self.elites[j]
                crossed = crossover_modules(
                    parent_a.candidate.dsl_source,
                    parent_b.candidate.dsl_source,
                )
                if crossed is not None:
                    new_candidates.append(
                        PTPDiscoveryCandidate(
                            dsl_source=crossed,
                            origin="crossover",
                            parent_ids=[
                                parent_a.result.candidate_id,
                                parent_b.result.candidate_id,
                            ],
                        )
                    )

        # Truncate to population size.
        return new_candidates[: self.population_size]

