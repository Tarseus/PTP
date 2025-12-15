from __future__ import annotations

"""
PTP-Discovery: LLM-guided evolutionary search for preference teaching strategies.

Usage (from repo root, after installing dependencies):

    pip install openai python-dotenv
    python -m ptp_discovery.run_llm_search \\
        --generations 5 \\
        --population 8 \\
        --hf-steps 200 \\
        --device cuda

Environment variables (typically set in .env):

    OPENAI_API_KEY      # required
    OPENAI_BASE_URL     # optional, default: https://api.openai.com/v1
    OPENAI_MODEL        # optional, default: gpt-4.1
"""

import argparse
import json
import logging
import os
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from fitness.ptp_high_fidelity import HighFidelityConfig
from ptp_discovery.problem import PTPDiscoveryCandidate, PTPDiscoveryResult
from ptp_discovery.search import PTPDiscoverySearch, EliteRecord
from ptp_discovery.ptp_best import PTP_BEST_DSL


LOGGER = logging.getLogger("ptp_discovery.run_llm_search")


def _load_env() -> None:
    """Load .env if present and validate required keys."""

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please create a .env file at the "
            "repo root with OPENAI_API_KEY=sk-... and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL."
        )


def _make_openai_client() -> OpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _read_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json_object(text: str) -> str:
    """Extract the first top-level JSON object substring from a text.

    This makes the pipeline robust to models that wrap the JSON in markdown
    code fences or prepend explanations, while still enforcing strict JSON.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def generate_ptp_dsl(client: OpenAI, extra_hint: str = "", burn_in_json: str | None = None) -> str:
    """Call the LLM to generate a single PTP DSL JSON program."""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system_path = os.path.join(base_dir, "prompts", "ptp_discovery_system.txt")
    task_path = os.path.join(base_dir, "prompts", "ptp_discovery_task.txt")

    system_prompt = _read_prompt(system_path)
    task_prompt = _read_prompt(task_path)

    user_content = task_prompt
    if burn_in_json:
        user_content += "\n\nBURN_IN_OBJECTIVES_JSON:\n" + burn_in_json
    if extra_hint:
        user_content += "\n" + extra_hint

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content,
        },
    ]

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")

    LOGGER.info("Requesting PTP DSL from model=%s", model_name)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()

    # Extract and validate JSON.
    json_str = _extract_json_object(raw)
    dsl_obj: Any = json.loads(json_str)
    # Re-dump with stable formatting for reproducibility.
    return json.dumps(dsl_obj, indent=2, sort_keys=True)


def _build_mutation_hint(
    elites: Sequence[EliteRecord],
) -> str:
    """Construct an extra hint for LLM-guided mutation/crossover.

    This presents a small subset of elite candidates and their metrics,
    and asks the model to propose a new child DSL that mutates or
    recombines their design.
    """

    lines = []
    lines.append("\nYou are now performing evolutionary mutation/crossover "
                 "on existing PTP DSL programs.")
    lines.append("Below are some elite parent candidates with their metrics:")

    for rec in elites:
        cid = rec.result.candidate_id
        hf = rec.result.hf_score
        val = rec.result.validation_objective
        gen_pen = rec.result.generalization_penalty
        lines.append(f"\nParent candidate_id={cid}")
        lines.append(f"HF_score={hf:.6f}, validation_objective={val:.6f}, "
                     f"generalization_penalty={gen_pen:.6f}")
        lines.append("PTP_DSL_JSON:")
        lines.append(rec.candidate.dsl_source)

    lines.append(
        "\nUsing these parents as inspiration, propose a NEW child PTP DSL "
        "program that mutates or recombines their design.\n"
        "- You may change anchors, build_preferences, and weight freely.\n"
        "- You may mix ideas from multiple parents.\n"
        "- Do NOT include any commentary; output ONLY the JSON object."
    )

    return "\n".join(lines)


def _llm_mutate_candidates(
    client: OpenAI,
    elites: Sequence[EliteRecord],
    population_size: int,
    burn_in_json: str | None,
) -> list[PTPDiscoveryCandidate]:
    """Use the LLM to generate mutated/crossover candidates from elites."""

    if not elites or population_size <= 0:
        return []

    # To keep prompts manageable, only show a small number of parents.
    max_parents_in_prompt = 4
    parents_for_prompt = list(elites[:max_parents_in_prompt])

    new_candidates: list[PTPDiscoveryCandidate] = []

    # Generate up to population_size children, cycling over the elite pool.
    while len(new_candidates) < population_size:
        mutation_hint = _build_mutation_hint(parents_for_prompt)
        try:
            dsl = generate_ptp_dsl(client, extra_hint=mutation_hint, burn_in_json=burn_in_json)
            parent_ids = [rec.result.candidate_id for rec in parents_for_prompt]
            new_candidates.append(
                PTPDiscoveryCandidate(
                    dsl_source=dsl,
                    origin="llm_mutation",
                    parent_ids=parent_ids,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to generate mutated candidate via LLM: %s", exc)
            break

    return new_candidates


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-guided evolutionary search over PTP preference teaching strategies.",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of evolutionary generations to run (excluding gen 0).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=8,
        help="Target population size per generation (after mutation + LLM).",
    )
    parser.add_argument(
        "--elites",
        type=int,
        default=4,
        help="Number of elites to keep between generations.",
    )
    parser.add_argument(
        "--init-llm",
        type=int,
        default=2,
        help="Number of LLM-generated candidates in the initial population (in addition to the baseline).",
    )
    parser.add_argument(
        "--hf-steps",
        type=int,
        default=200,
        help="Short-run training steps used in high-fidelity evaluation.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=20,
        help="Problem size used during short-run training.",
    )
    parser.add_argument(
        "--valid-sizes",
        type=int,
        nargs="+",
        default=[100],
        help="Problem sizes used for generalization validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string for training/evaluation (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for HF evaluation.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="ptp_logs",
        help="Directory where candidate programs and HF metrics will be stored.",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    _load_env()
    client = _make_openai_client()

    parser = _build_arg_parser()
    args = parser.parse_args()

    hf_cfg = HighFidelityConfig(
        problem="tsp",
        hf_steps=args.hf_steps,
        train_problem_size=args.train_size,
        valid_problem_sizes=tuple(args.valid_sizes),
        device=args.device,
        seed=args.seed,
    )

    # Burn-in: evaluate the baseline PTP_BEST program once to obtain
    # reference metrics that are fed into the LLM context.
    from fitness.ptp_high_fidelity import evaluate_ptp_dsl_high_fidelity

    baseline_metrics = evaluate_ptp_dsl_high_fidelity(PTP_BEST_DSL, hf_cfg)
    burn_in_objectives = [
        {
            "name": "PTP_BEST",
            "type": "ptp_dsl_program",
            "dsl_source": PTP_BEST_DSL,
            "metrics": baseline_metrics,
        }
    ]
    burn_in_json = json.dumps(burn_in_objectives, indent=2)

    search = PTPDiscoverySearch(
        hf_config=hf_cfg,
        log_dir=args.log_dir,
        population_size=args.population,
        elite_size=args.elites,
    )

    # Generation 0: baseline + a few LLM seeds.
    init_candidates = [
        PTPDiscoveryCandidate(dsl_source=PTP_BEST_DSL, origin="baseline"),
    ]
    for _ in range(max(0, args.init_llm)):
        try:
            dsl = generate_ptp_dsl(client, burn_in_json=burn_in_json)
            init_candidates.append(PTPDiscoveryCandidate(dsl_source=dsl, origin="llm"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to generate initial LLM candidate: %s", exc)

    LOGGER.info("Evaluating generation 0 with %d candidates", len(init_candidates))
    elites = search.evaluate_generation(init_candidates)
    best = elites[0]
    LOGGER.info(
        "Gen 0 best HF_score=%.6f candidate_id=%s",
        best.result.hf_score,
        best.result.candidate_id,
    )

    # Subsequent generations: LLM-driven mutation/crossover + occasional fresh seeds.
    for gen in range(1, args.generations + 1):
        LOGGER.info("=== Generation %d ===", gen)

        def _mutate_fn(elites: Sequence[EliteRecord], population_size: int):
            return _llm_mutate_candidates(client, elites, population_size, burn_in_json)

        mutated = search.propose_mutations(_mutate_fn)
        LOGGER.info("Generated %d mutated/crossover candidates via LLM", len(mutated))

        # Top up the population with fresh LLM candidates if needed.
        needed = max(0, args.population - len(mutated))
        for _ in range(needed):
            try:
                dsl = generate_ptp_dsl(client, burn_in_json=burn_in_json)
                mutated.append(PTPDiscoveryCandidate(dsl_source=dsl, origin="llm"))
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to generate LLM candidate in gen %d: %s", gen, exc)

        LOGGER.info("Evaluating generation %d with %d candidates", gen, len(mutated))
        elites = search.evaluate_generation(mutated)
        best = elites[0]
        LOGGER.info(
            "Gen %d best HF_score=%.6f candidate_id=%s",
            gen,
            best.result.hf_score,
            best.result.candidate_id,
        )

    LOGGER.info(
        "Search complete. Best candidate across all generations: HF_score=%.6f, id=%s",
        best.result.hf_score,
        best.result.candidate_id,
    )
    LOGGER.info("Artifacts saved under: %s", os.path.abspath(args.log_dir))


if __name__ == "__main__":
    main()
