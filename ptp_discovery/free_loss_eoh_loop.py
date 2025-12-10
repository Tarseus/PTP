from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml

from fitness.free_loss_fidelity import (
    FreeLossFidelityConfig,
    evaluate_free_loss_candidate,
)
from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed, _evaluate_tsp_model
from ptp_discovery.free_loss_compiler import CompileError
from ptp_discovery.free_loss_gates import (
    DynamicGateResult,
    StaticGateResult,
    run_dynamic_gates,
    run_static_gates,
)
from ptp_discovery.free_loss_ir import FreeLossIR
from ptp_discovery.free_loss_llm_ops import (
    compile_free_loss_candidate,
    crossover_free_loss,
    generate_free_loss_candidate,
    mutate_free_loss,
)

from TSPEnv import TSPEnv  # type: ignore  # noqa: E402
from TSPModel import TSPModel  # type: ignore  # noqa: E402
from utils.utils import AverageMeter  # type: ignore  # noqa: E402
from torch.optim import Adam


LOGGER = logging.getLogger("ptp_discovery.free_loss_eoh")


def _timestamp_dir(root: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _dump_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            # Use UTF-8 and allow non-ASCII characters for readability.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _train_one_batch_with_po(
    env: TSPEnv,
    model: TSPModel,
    optimizer: Adam,
    cfg: HighFidelityConfig,
) -> Tuple[float, float]:
    """Single training batch using the original POMO PO loss (po_loss)."""

    batch_size = cfg.train_batch_size
    aug_factor = 1

    model.train()
    env.load_problems(batch_size // aug_factor, aug_factor=aug_factor)

    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    prob_list = torch.zeros(
        size=(batch_size, env.pomo_size, 0),
        device=env.problems.device,
    )

    state, reward, done = env.pre_step()
    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

    # reward: (batch, pomo) with negative tour length.
    # This matches the original POMO implementation.
    preference = reward[:, :, None] > reward[:, None, :]
    log_prob = torch.log(prob_list + 1e-8).sum(dim=2)
    log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]

    alpha = float(cfg.alpha)
    pf_log = torch.log(torch.sigmoid(alpha * log_prob_pair))
    loss = -torch.mean(pf_log * preference)

    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return score_mean.item(), float(loss.item())


def evaluate_po_baseline(cfg: HighFidelityConfig) -> Dict[str, Any]:
    """Short-run HF-style evaluation using the original POMO po_loss."""

    if cfg.problem != "tsp":
        raise NotImplementedError(
            "evaluate_po_baseline currently supports only problem='tsp'."
        )

    _set_seed(cfg.seed)

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 50,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }

    env = TSPEnv(
        problem_size=cfg.train_problem_size,
        pomo_size=cfg.pomo_size,
        device=str(device),
    )

    model = TSPModel(**model_params).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    score_meter = AverageMeter()
    loss_meter = AverageMeter()

    LOGGER.info(
        "Baseline PO training: steps=%d, train_problem_size=%d, pomo_size=%d, "
        "batch_size=%d, device=%s",
        cfg.hf_steps,
        cfg.train_problem_size,
        cfg.pomo_size,
        cfg.train_batch_size,
        str(device),
    )

    log_interval = max(cfg.hf_steps // 10, 1)

    for step in range(cfg.hf_steps):
        score, loss = _train_one_batch_with_po(env, model, optimizer, cfg)
        score_meter.update(score)
        loss_meter.update(loss)

        if (step + 1) % log_interval == 0 or step == 0:
            LOGGER.info(
                "Baseline PO step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f)",
                step + 1,
                cfg.hf_steps,
                score,
                float(score_meter.avg),
                loss,
                float(loss_meter.avg),
            )

    main_valid_obj = _evaluate_tsp_model(
        model=model,
        problem_size=cfg.train_problem_size,
        pomo_size=cfg.pomo_size,
        device=device,
        num_episodes=cfg.num_validation_episodes,
        batch_size=cfg.validation_batch_size,
    )

    gen_objectives: Dict[int, float] = {}
    for size in cfg.valid_problem_sizes:
        size_int = int(size)
        gen_obj = _evaluate_tsp_model(
            model=model,
            problem_size=size_int,
            pomo_size=cfg.pomo_size,
            device=device,
            num_episodes=cfg.num_validation_episodes,
            batch_size=cfg.validation_batch_size,
        )
        gen_objectives[size_int] = gen_obj

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    hf_score = main_valid_obj + cfg.generalization_penalty_weight * generalization_penalty

    return {
        "hf_score": hf_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "config": {
            "hf": cfg.__dict__,
            "baseline_type": "po_loss",
        },
    }


def run_free_loss_eoh(config_path: str, **overrides: Any) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    cfg_yaml.update({k: v for k, v in overrides.items() if v is not None})

    LOGGER.info("Starting free loss EoH search with config=%s", config_path)

    seed = int(cfg_yaml.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generations = int(cfg_yaml.get("generations", 1))
    population_size = int(cfg_yaml.get("population_size", 8))
    elite_size = int(cfg_yaml.get("elite_size", 4))
    init_llm = int(cfg_yaml.get("init_llm", 4))

    hf_cfg = HighFidelityConfig(
        problem=cfg_yaml.get("problem", "tsp"),
        hf_steps=int(cfg_yaml.get("f1_steps", 32)),
        train_problem_size=int(cfg_yaml.get("train_problem_size", 20)),
        valid_problem_sizes=tuple(int(v) for v in cfg_yaml.get("valid_problem_sizes", [100])),
        train_batch_size=int(cfg_yaml.get("train_batch_size", 64)),
        pomo_size=int(cfg_yaml.get("pomo_size", 64)),
        learning_rate=float(cfg_yaml.get("learning_rate", 3e-4)),
        weight_decay=float(cfg_yaml.get("weight_decay", 1e-6)),
        alpha=float(cfg_yaml.get("alpha", 0.05)),
        device=str(cfg_yaml.get("device", "cuda")),
        seed=seed,
        num_validation_episodes=int(cfg_yaml.get("num_validation_episodes", 128)),
        validation_batch_size=int(cfg_yaml.get("validation_batch_size", 64)),
        generalization_penalty_weight=float(cfg_yaml.get("generalization_penalty_weight", 1.0)),
        pool_version="v0",
    )

    free_cfg = FreeLossFidelityConfig(
        hf=hf_cfg,
        f1_steps=int(cfg_yaml.get("f1_steps", 32)),
        f2_steps=int(cfg_yaml.get("f2_steps", 0)),
        f3_enabled=bool(cfg_yaml.get("f3_enabled", False)),
    )

    baseline_hf_score: float | None = None

    # Baseline: evaluate the original POMO po_loss once, using the same HF
    # configuration. This provides a reference score before searching over
    # free-form preference losses.
    try:
        baseline = evaluate_po_baseline(hf_cfg)
        baseline_hf_score = float(baseline["hf_score"])
        LOGGER.info(
            "Baseline po_loss: hf_score=%.6f, validation_objective=%.6f, gen_penalty=%.6f",
            baseline["hf_score"],
            baseline["validation_objective"],
            baseline["generalization_penalty"],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to evaluate baseline po_loss: %s", exc)

    operator_whitelist = list(cfg_yaml.get("operator_whitelist", []))
    prompts = cfg_yaml.get("prompts", {}) or {}
    gen_prompt = prompts.get("generation")
    crossover_prompt = prompts.get("crossover")
    mutation_prompt = prompts.get("mutation")

    out_root = cfg_yaml.get("output_root", "runs/free_loss_discovery")
    run_dir = _timestamp_dir(out_root)
    LOGGER.info("Run directory: %s", os.path.abspath(run_dir))

    candidates_log: List[Dict[str, Any]] = []
    gates_log: List[Dict[str, Any]] = []
    fitness_log: List[Dict[str, Any]] = []

    elites: List[Dict[str, Any]] = []

    for gen in range(generations):
        LOGGER.info("=== Generation %d/%d ===", gen, generations - 1)
        population: List[FreeLossIR] = []
        if gen == 0:
            LOGGER.info("Generating initial population with %d LLM candidates", init_llm)
            for _ in range(init_llm):
                ir = generate_free_loss_candidate(gen_prompt, operator_whitelist=operator_whitelist)
                population.append(ir)
        else:
            parents = sorted(elites, key=lambda e: e["fitness"]["hf_like_score"])[:elite_size]
            parent_irs = [p["ir"] for p in parents]
            LOGGER.info(
                "Generating population via crossover/mutation: size=%d, elite_size=%d, available_elites=%d",
                population_size,
                elite_size,
                len(parent_irs),
            )
            for _ in range(population_size):
                if len(parent_irs) >= 2 and crossover_prompt:
                    ir = crossover_free_loss(crossover_prompt, parent_irs[:2])
                elif parent_irs and mutation_prompt:
                    ir = mutate_free_loss(mutation_prompt, parent_irs[0])
                else:
                    ir = generate_free_loss_candidate(gen_prompt, operator_whitelist=operator_whitelist)
                population.append(ir)

        LOGGER.info("Population size for generation %d: %d", gen, len(population))
        gen_elites: List[Dict[str, Any]] = []

        static_fail = 0
        dynamic_fail = 0
        evaluated = 0

        for idx, ir in enumerate(population):
            static_res: StaticGateResult
            static_res = run_static_gates(ir, operator_whitelist=operator_whitelist)
            gate_entry: Dict[str, Any] = {
                "generation": gen,
                "index": idx,
                "ir": asdict(ir),
                "static_ok": static_res.ok,
                "static_reason": static_res.reason,
            }

            if not static_res.ok:
                static_fail += 1
                gates_log.append(gate_entry)
                continue

            try:
                compiled = compile_free_loss_candidate(ir, operator_whitelist=operator_whitelist)
            except CompileError as exc:
                gate_entry["dynamic_ok"] = False
                gate_entry["dynamic_reason"] = f"compile_error: {exc}"
                gates_log.append(gate_entry)
                continue

            model = TSPModel(
                embedding_dim=128,
                sqrt_embedding_dim=128 ** 0.5,
                encoder_layer_num=1,
                decoder_layer_num=1,
                qkv_dim=16,
                head_num=8,
                logit_clipping=50,
                ff_hidden_dim=128,
                eval_type="argmax",
            )

            dummy_batch = {
                "cost_a": torch.zeros(16),
                "cost_b": torch.ones(16),
                "log_prob_w": torch.zeros(16),
                "log_prob_l": torch.zeros(16),
                "weight": torch.ones(16),
            }

            dyn_res: DynamicGateResult
            dyn_res = run_dynamic_gates(
                compiled,
                batch=dummy_batch,
                model=model,
                grad_norm_max=float(cfg_yaml.get("grad_norm_max", 10.0)),
                loss_soft_min=float(cfg_yaml.get("loss_soft_min", -5.0)),
                loss_soft_max=float(cfg_yaml.get("loss_soft_max", 5.0)),
            )

            gate_entry.update(
                {
                    "dynamic_ok": dyn_res.ok,
                    "dynamic_reason": dyn_res.reason,
                    "loss_value": dyn_res.loss_value,
                    "grad_norm": dyn_res.grad_norm,
                }
            )
            gates_log.append(gate_entry)

            if not dyn_res.ok:
                dynamic_fail += 1
                continue

            fitness = evaluate_free_loss_candidate(compiled, free_cfg)
            evaluated += 1

            LOGGER.info(
                "Gen %d cand %d: hf_like_score=%.6f, validation_objective=%.6f",
                gen,
                idx,
                float(fitness["hf_like_score"]),
                float(fitness["validation_objective"]),
            )

            cand_entry_log = {
                "generation": gen,
                "index": idx,
                "ir": asdict(ir),
                "fitness": fitness,
            }
            candidates_log.append(cand_entry_log)
            fitness_log.append(
                {
                    "generation": gen,
                    "index": idx,
                    "hf_like_score": fitness["hf_like_score"],
                    "validation_objective": fitness["validation_objective"],
                }
            )
            elite_entry = {
                "generation": gen,
                "index": idx,
                "ir": ir,
                "fitness": fitness,
            }
            gen_elites.append(elite_entry)

        def _elite_key(entry: Dict[str, Any]) -> Tuple[float, float, float]:
            score = float(entry["fitness"]["hf_like_score"])
            pair_count = float(entry["fitness"].get("pair_count", 0) or 0)
            if baseline_hf_score is None:
                # Fallback: purely score-based, with pair_count as a tie-breaker.
                return (0.0, score, pair_count)
            # Prefer candidates that beat the baseline (lower score).
            better = score <= baseline_hf_score
            flag = 0.0 if better else 1.0
            # When better than baseline, fewer pairs is preferred; otherwise ignore pair_count.
            effective_pairs = pair_count if better else 0.0
            return (flag, score, effective_pairs)

        gen_elites.sort(key=_elite_key)
        LOGGER.info(
            "Generation %d summary: static_fail=%d, dynamic_fail=%d, evaluated=%d, new_elites=%d",
            gen,
            static_fail,
            dynamic_fail,
            evaluated,
            len(gen_elites),
        )
        elites.extend(gen_elites)
        elites.sort(key=_elite_key)
        elites = elites[:elite_size]

    _dump_jsonl(os.path.join(run_dir, "candidates.jsonl"), candidates_log)
    _dump_jsonl(os.path.join(run_dir, "gate_reports.jsonl"), gates_log)
    _dump_jsonl(os.path.join(run_dir, "fitness_scores.jsonl"), fitness_log)

    if elites:
        best = elites[0]
        best_path = os.path.join(run_dir, "best_candidate.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        LOGGER.info(
            "Search complete. Best hf_like_score=%.6f (generation=%d, index=%d)",
            best["fitness"]["hf_like_score"],
            best["generation"],
            best["index"],
        )
    else:
        LOGGER.info("Search complete. No candidate passed dynamic gates; no elites selected.")
