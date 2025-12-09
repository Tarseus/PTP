from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from fitness.ptp_high_fidelity import HighFidelityConfig, evaluate_ptp_dsl_high_fidelity
from .problem import PTPDiscoveryCandidate, PTPDiscoveryResult, PTPDiscoveryProblem


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

    def propose_mutations(
        self,
        mutate_fn: Callable[[Sequence[EliteRecord], int], List[PTPDiscoveryCandidate]],
    ) -> List[PTPDiscoveryCandidate]:
        """Generate new candidates using an external mutation function.

        This is intentionally generic so that an external controller (e.g. an
        LLM-based orchestrator) can implement mutation and crossover however it
        likes, given access to the current elite pool.
        """

        if not self.elites:
            return []

        return mutate_fn(self.elites, self.population_size)

