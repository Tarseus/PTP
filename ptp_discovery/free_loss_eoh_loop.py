from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List

import torch
import yaml

from fitness.free_loss_fidelity import (
    FreeLossFidelityConfig,
    evaluate_free_loss_candidate,
)
from fitness.ptp_high_fidelity import HighFidelityConfig
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

from TSPModel import TSPModel  # type: ignore  # noqa: E402


def _timestamp_dir(root: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _dump_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def run_free_loss_eoh(config_path: str, **overrides: Any) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    cfg_yaml.update({k: v for k, v in overrides.items() if v is not None})

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

    operator_whitelist = list(cfg_yaml.get("operator_whitelist", []))
    prompts = cfg_yaml.get("prompts", {}) or {}
    gen_prompt = prompts.get("generation")
    crossover_prompt = prompts.get("crossover")
    mutation_prompt = prompts.get("mutation")

    out_root = cfg_yaml.get("output_root", "runs/free_loss_discovery")
    run_dir = _timestamp_dir(out_root)

    candidates_log: List[Dict[str, Any]] = []
    gates_log: List[Dict[str, Any]] = []
    fitness_log: List[Dict[str, Any]] = []

    elites: List[Dict[str, Any]] = []

    for gen in range(generations):
        population: List[FreeLossIR] = []
        if gen == 0:
            for _ in range(init_llm):
                ir = generate_free_loss_candidate(gen_prompt, operator_whitelist=operator_whitelist)
                population.append(ir)
        else:
            parents = sorted(elites, key=lambda e: e["fitness"]["hf_like_score"])[:elite_size]
            parent_irs = [p["ir"] for p in parents]
            for _ in range(population_size):
                if len(parent_irs) >= 2 and crossover_prompt:
                    ir = crossover_free_loss(crossover_prompt, parent_irs[:2])
                elif parent_irs and mutation_prompt:
                    ir = mutate_free_loss(mutation_prompt, parent_irs[0])
                else:
                    ir = generate_free_loss_candidate(gen_prompt, operator_whitelist=operator_whitelist)
                population.append(ir)

        gen_elites: List[Dict[str, Any]] = []

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
                continue

            fitness = evaluate_free_loss_candidate(compiled, free_cfg)

            cand_entry = {
                "generation": gen,
                "index": idx,
                "ir": asdict(ir),
                "fitness": fitness,
            }
            candidates_log.append(cand_entry)
            fitness_log.append(
                {
                    "generation": gen,
                    "index": idx,
                    "hf_like_score": fitness["hf_like_score"],
                    "validation_objective": fitness["validation_objective"],
                }
            )
            gen_elites.append(cand_entry)

        gen_elites.sort(key=lambda e: e["fitness"]["hf_like_score"])
        elites.extend(gen_elites)
        elites.sort(key=lambda e: e["fitness"]["hf_like_score"])
        elites = elites[:elite_size]

    _dump_jsonl(os.path.join(run_dir, "candidates.jsonl"), candidates_log)
    _dump_jsonl(os.path.join(run_dir, "gate_reports.jsonl"), gates_log)
    _dump_jsonl(os.path.join(run_dir, "fitness_scores.jsonl"), fitness_log)

    if elites:
        best = elites[0]
        best_path = os.path.join(run_dir, "best_candidate.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

