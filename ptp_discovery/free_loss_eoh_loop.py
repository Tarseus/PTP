from __future__ import annotations

import json
import os
import time
import logging
import multiprocessing as mp
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
from ptp_discovery.free_loss_ir import FreeLossIR, ir_from_json
from ptp_discovery.free_loss_llm_ops import (
    compile_free_loss_candidate,
    crossover_free_loss,
    generate_free_loss_candidate,
    mutate_free_loss,
    repair_expects_with_prompt,
)

from TSPEnv import TSPEnv  # type: ignore  # noqa: E402
from TSPModel import TSPModel  # type: ignore  # noqa: E402
from utils.utils import AverageMeter  # type: ignore  # noqa: E402
from torch.optim import Adam


LOGGER = logging.getLogger("ptp_discovery.free_loss_eoh")


def _get_available_devices(base_device: str) -> List[str]:
    """Return a list of logical device strings for parallel evaluation.

    - If base_device is 'cuda', expose all visible CUDA devices as
      ['cuda:0', 'cuda:1', ...].
    - If base_device has an explicit index (e.g., 'cuda:3'), keep it as a
      single-device list.
    - For CPU or unknown strings, fall back to [base_device].
    """

    base_device = str(base_device)
    if base_device.startswith("cuda:"):
        return [base_device]
    if base_device == "cuda" and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count <= 0:
            return [base_device]
        return [f"cuda:{i}" for i in range(count)]
    return [base_device]


def _worker_evaluate_candidate(args: Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    str,
    List[str],
    str,
    int,
    int,
]) -> Tuple[int, Dict[str, Any]]:
    """Worker process: compile and evaluate a single candidate loss.

    To keep the top-level discovery log readable, this worker redirects
    training logs for each candidate into a dedicated file under the run
    directory instead of emitting them to stdout/stderr.
    """

    (
        ir_payload,
        hf_cfg_dict,
        free_cfg_dict,
        device_str,
        operator_whitelist,
        run_dir,
        gen,
        idx,
    ) = args

    # Reduce noise on stdout/stderr from worker processes.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    # Route free-loss training logs to a per-candidate file.
    log_path = os.path.join(run_dir, f"gen{gen:03d}_cand{idx:03d}.log")
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    fl_logger = logging.getLogger("fitness.free_loss_fidelity")
    fl_logger.handlers = []
    fl_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    fl_logger.addHandler(file_handler)

    # Reconstruct configs for this worker and override device.
    hf_cfg = HighFidelityConfig(**hf_cfg_dict)
    hf_cfg.device = device_str
    free_cfg = FreeLossFidelityConfig(
        hf=hf_cfg,
        f1_steps=int(free_cfg_dict.get("f1_steps", 32)),
        f2_steps=int(free_cfg_dict.get("f2_steps", 0)),
        f3_enabled=bool(free_cfg_dict.get("f3_enabled", False)),
    )

    # Reconstruct IR and compiled loss in the worker.
    ir = ir_from_json(ir_payload)
    compiled = compile_free_loss_candidate(ir, operator_whitelist=operator_whitelist)

    fitness = evaluate_free_loss_candidate(compiled, free_cfg)
    return idx, fitness


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

    t_init_start = time.perf_counter()
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
    t_init_end = time.perf_counter()

    score_meter = AverageMeter()
    loss_meter = AverageMeter()

    LOGGER.info(
        "Baseline PO training: steps=%d, train_problem_size=%d, pomo_size=%d, "
        "batch_size=%d, device=%s, init_time=%.3fs",
        cfg.hf_steps,
        cfg.train_problem_size,
        cfg.pomo_size,
        cfg.train_batch_size,
        str(device),
        t_init_end - t_init_start,
    )

    log_interval = max(cfg.hf_steps // 20, 1)

    t_train_start = time.perf_counter()
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
    t_train_end = time.perf_counter()

    t_eval_start = time.perf_counter()
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
    t_eval_end = time.perf_counter()

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    hf_score = main_valid_obj + cfg.generalization_penalty_weight * generalization_penalty

    LOGGER.info(
        "Baseline PO timing: init=%.3fs, train=%.3fs, eval=%.3fs",
        t_init_end - t_init_start,
        t_train_end - t_train_start,
        t_eval_end - t_eval_start,
    )

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
    _set_seed(seed)

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
    hf_cfg_dict: Dict[str, Any] = dict(hf_cfg.__dict__)
    free_cfg_dict: Dict[str, Any] = {
        "f1_steps": free_cfg.f1_steps,
        "f2_steps": free_cfg.f2_steps,
        "f3_enabled": free_cfg.f3_enabled,
    }

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
    expects_repair_prompt = prompts.get("expects_repair")

    out_root = cfg_yaml.get("output_root", "runs/free_loss_discovery")
    run_dir = _timestamp_dir(out_root)
    LOGGER.info("Run directory: %s", os.path.abspath(run_dir))

    candidates_log: List[Dict[str, Any]] = []
    gates_log: List[Dict[str, Any]] = []
    fitness_log: List[Dict[str, Any]] = []

    elites: List[Dict[str, Any]] = []

    def _maybe_repair_expects(ir: FreeLossIR) -> FreeLossIR:
        if not expects_repair_prompt:
            return ir
        expects = ir.implementation_hint.expects or []
        # Only call the repair prompt when we already have a list; this
        # is meant to normalize names, not to infer them from scratch.
        if not isinstance(expects, (list, tuple)) or not expects:
            return ir
        try:
            return repair_expects_with_prompt(expects_repair_prompt, ir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to repair expects via LLM: %s", exc)
            return ir

    for gen in range(generations):
        LOGGER.info("=== Generation %d/%d ===", gen, generations - 1)
        population: List[FreeLossIR] = []
        if gen == 0:
            LOGGER.info("Generating initial population with %d LLM candidates", init_llm)
            for _ in range(init_llm):
                ir = generate_free_loss_candidate(gen_prompt, operator_whitelist=operator_whitelist)
                population.append(_maybe_repair_expects(ir))
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
                population.append(_maybe_repair_expects(ir))

        LOGGER.info("Population size for generation %d: %d", gen, len(population))
        gen_elites: List[Dict[str, Any]] = []

        static_fail = 0
        dynamic_fail = 0
        evaluated = 0

        # Collect candidates that pass all gates and evaluate them in
        # parallel across available devices.
        eval_jobs: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, List[str], str, int, int]] = []
        eval_candidates: Dict[int, FreeLossIR] = {}

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
                LOGGER.warning(
                    "Dynamic gates failed for gen=%d, idx=%d, name=%s: reason=%s, "
                    "loss_value=%s, grad_norm=%s",
                    gen,
                    idx,
                    ir.name,
                    dyn_res.reason,
                    str(dyn_res.loss_value),
                    str(dyn_res.grad_norm),
                )
                continue

            # Candidate passes all gates; queue it for evaluation.
            eval_candidates[idx] = ir
            ir_payload = asdict(ir)
            eval_jobs.append(
                (
                    ir_payload,
                    hf_cfg_dict,
                    free_cfg_dict,
                    "",  # device to be filled after we know available devices
                    operator_whitelist,
                    run_dir,
                    gen,
                    idx,
                )
            )

        # Evaluate all surviving candidates in parallel across GPUs / devices.
        results: List[Tuple[int, Dict[str, Any]]] = []
        if eval_jobs:
            devices = _get_available_devices(hf_cfg.device)
            if not devices:
                devices = [hf_cfg.device]

            # Assign one candidate per device in a round-robin fashion.
            jobs_with_devices: List[Tuple[
                Dict[str, Any],
                Dict[str, Any],
                Dict[str, Any],
                str,
                List[str],
                str,
                int,
                int,
            ]] = []
            for j_idx, job in enumerate(eval_jobs):
                device_str = devices[j_idx % len(devices)]
                job_with_dev = list(job)
                job_with_dev[3] = device_str
                jobs_with_devices.append(tuple(job_with_dev))  # type: ignore[arg-type]

            ctx = mp.get_context("spawn")
            num_workers = min(len(devices), len(jobs_with_devices))
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(_worker_evaluate_candidate, jobs_with_devices)

        # Integrate evaluation results back into the evolutionary loop.
        for idx, fitness in sorted(results, key=lambda x: x[0]):
            evaluated += 1
            ir = eval_candidates[idx]

            hf_like_score = float(fitness["hf_like_score"])
            better_than_baseline = None
            if baseline_hf_score is not None:
                # Lower score is better.
                better_than_baseline = hf_like_score <= baseline_hf_score

            LOGGER.info(
                "Gen %d cand %d: hf_like_score=%.6f, validation_objective=%.6f, "
                "baseline=%.6f, better_than_baseline=%s",
                gen,
                idx,
                hf_like_score,
                float(fitness["validation_objective"]),
                float(baseline_hf_score) if baseline_hf_score is not None else float("nan"),
                str(better_than_baseline),
            )

            cand_entry_log = {
                "generation": gen,
                "index": idx,
                "ir": asdict(ir),
                "fitness": fitness,
                "better_than_baseline": better_than_baseline,
            }
            candidates_log.append(cand_entry_log)
            fitness_log.append(
                {
                    "generation": gen,
                    "index": idx,
                    "hf_like_score": fitness["hf_like_score"],
                    "validation_objective": fitness["validation_objective"],
                    "baseline_hf_score": baseline_hf_score,
                    "better_than_baseline": better_than_baseline,
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
        # Make sure `ir` is JSON-serializable (convert FreeLossIR dataclass to dict).
        best_serializable = dict(best)
        ir_value = best_serializable.get("ir")
        if isinstance(ir_value, FreeLossIR):
            best_serializable["ir"] = asdict(ir_value)
        best_path = os.path.join(run_dir, "best_candidate.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_serializable, f, indent=2, ensure_ascii=False)
        LOGGER.info(
            "Search complete. Best hf_like_score=%.6f (generation=%d, index=%d)",
            best["fitness"]["hf_like_score"],
            best["generation"],
            best["index"],
        )
    else:
        LOGGER.info("Search complete. No candidate passed dynamic gates; no elites selected.")
