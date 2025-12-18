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
from fitness.ptp_high_fidelity import (
    HighFidelityConfig,
    _set_seed,
    _evaluate_tsp_model,
    get_total_hf_train_steps,
)
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
    repair_free_loss,
    repair_expects_with_prompt,
)

from TSPEnv import TSPEnv  # type: ignore  # noqa: E402
from TSPModel import TSPModel  # type: ignore  # noqa: E402
from utils.utils import AverageMeter  # type: ignore  # noqa: E402
from torch.optim import Adam


LOGGER = logging.getLogger("ptp_discovery.free_loss_eoh")


def _classify_failure(stage: str, reason: str) -> str:
    """Map free-loss gate failures to coarse error codes.

    These codes are fed into LLM repair / mutation prompts so that the
    model can learn which failure modes to avoid.
    """

    msg = (reason or "").lower()
    stage = stage.lower()

    if stage == "static":
        if "missing name" in msg:
            return "E_STATIC_MISSING_NAME"
        if "missing pseudocode" in msg:
            return "E_STATIC_MISSING_PSEUDOCODE"
        if "operators_used must be non-empty" in msg:
            return "E_STATIC_EMPTY_OPERATORS"
        if "non-whitelisted operators" in msg:
            return "E_OPERATOR_VIOLATION"
        if "returns must describe a scalar" in msg:
            return "E_EXPECTS_RETURNS_MISMATCH"
        if "hyperparameter" in msg and "non-finite" in msg:
            return "E_STATIC_NON_FINITE_HYPERPARAM"
        return "E_STATIC_OTHER"

    if stage == "compile":
        if "failed to parse json" in msg or "no json object found" in msg:
            return "E_JSON_PARSE"
        return "E_COMPILE_ERROR"

    if stage == "dynamic":
        if "loss is not finite" in msg:
            return "E_RUNTIME_NAN_LOSS"
        if "nan/inf in gradients" in msg:
            return "E_RUNTIME_NAN_GRAD"
        if "backward_error" in msg:
            return "E_BACKWARD_ERROR"
        if "forward_error" in msg:
            return "E_FORWARD_ERROR"
        if "grad_norm" in msg and "exceeds max" in msg:
            return "E_GRAD_EXPLODE"
        if "outside soft range" in msg or "soft range" in msg:
            return "E_LOSS_OUT_OF_RANGE"
        return "E_DYNAMIC_OTHER"

    return "E_UNKNOWN"


def _build_failure_payload(
    *,
    generation: int,
    index: int,
    attempt: int,
    stage: str,
    code: str,
    message: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generation": int(generation),
        "index": int(index),
        "attempt": int(attempt),
        "stage": stage,
        "code": code,
        "message": message,
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def _build_global_feedback(
    *,
    elites: List[Dict[str, Any]],
    gates_log: List[Dict[str, Any]],
    burn_in_objectives: List[Dict[str, Any]],
    baseline_hf_score: float | None,
    max_elites: int = 8,
    max_failures: int = 64,
) -> Dict[str, Any]:
    """Summarize recent search history for LLM prompts.

    Includes:
    - burn-in objectives (e.g., baseline po_loss)
    - a small set of best elites with metrics and IR hints
    - a histogram of recent gate failures by error code
    """

    # Summarize elites (best-performing candidates across generations).
    sorted_elites = sorted(
        elites,
        key=lambda e: float(e["fitness"]["hf_like_score"]),
    )
    elite_summaries: List[Dict[str, Any]] = []
    for entry in sorted_elites[:max_elites]:
        ir: FreeLossIR = entry["ir"]
        fitness = entry["fitness"]
        elite_summaries.append(
            {
                "generation": int(entry.get("generation", -1)),
                "index": int(entry.get("index", -1)),
                "name": ir.name,
                "theoretical_basis": getattr(ir, "theoretical_basis", ""),
                "operators_used": list(ir.operators_used),
                "hyperparams": dict(ir.hyperparams),
                "hf_like_score": float(fitness.get("hf_like_score", float("inf"))),
                "validation_objective": float(fitness.get("validation_objective", float("inf"))),
                "generalization_penalty": float(fitness.get("generalization_penalty", 0.0)),
                "pair_count": int(fitness.get("pair_count", 0) or 0),
            }
        )

    # Summarize recent failures.
    recent_fail_entries = [
        e
        for e in gates_log
        if (not e.get("static_ok", True)) or (e.get("dynamic_ok") is False)
    ][-max_failures:]

    error_stats: Dict[str, int] = {}
    for e in recent_fail_entries:
        for key in ("static_error_code", "dynamic_error_code"):
            code = e.get(key)
            if not code:
                continue
            error_stats[code] = error_stats.get(code, 0) + 1

    failures_by_code = [
        {"code": code, "count": count}
        for code, count in sorted(error_stats.items(), key=lambda kv: kv[1], reverse=True)
    ]

    failure_examples: List[Dict[str, Any]] = []
    for e in recent_fail_entries:
        failure_examples.append(
            {
                "generation": int(e.get("generation", -1)),
                "index": int(e.get("index", -1)),
                "attempt": int(e.get("attempt", 0)),
                "static_ok": bool(e.get("static_ok", True)),
                "dynamic_ok": e.get("dynamic_ok"),
                "static_error_code": e.get("static_error_code"),
                "dynamic_error_code": e.get("dynamic_error_code"),
                "static_reason": e.get("static_reason"),
                "dynamic_reason": e.get("dynamic_reason"),
            }
        )

    # Suggest a coarse search mode for the LLM.
    suggested_mode = "explore"
    if elite_summaries and baseline_hf_score is not None:
        best_score = elite_summaries[0]["hf_like_score"]
        improvement = float(baseline_hf_score) - float(best_score)
        if improvement <= 0.0:
            suggested_mode = "explore"
        elif improvement < 0.1:
            suggested_mode = "combine"
        else:
            suggested_mode = "refine"

    return {
        "burn_in_objectives": burn_in_objectives,
        "recent_elites": elite_summaries,
        "recent_failures": {
            "by_code": failures_by_code,
            "examples": failure_examples,
        },
        "suggested_mode": suggested_mode,
    }


def _write_run_analysis(
    run_dir: str,
    *,
    baseline_hf_score: float | None,
    generations: int,
    population_size: int,
    gates_log: List[Dict[str, Any]],
    elites: List[Dict[str, Any]],
) -> None:
    """Emit a lightweight JSON summary for downstream analysis.

    This aggregates gate failure statistics and highlights the best
    discovered loss, so that external tools (or LLMs) can inspect
    the run without re-parsing all logs.
    """

    error_stats: Dict[str, int] = {}
    for entry in gates_log:
        for key in ("static_error_code", "dynamic_error_code"):
            code = entry.get(key)
            if not code:
                continue
            error_stats[code] = error_stats.get(code, 0) + 1

    failures_by_code = [
        {"code": code, "count": count}
        for code, count in sorted(error_stats.items(), key=lambda kv: kv[1], reverse=True)
    ]

    best_summary: Dict[str, Any] | None = None
    if elites:
        best = sorted(elites, key=lambda e: float(e["fitness"]["hf_like_score"]))[0]
        ir: FreeLossIR = best["ir"]
        fitness = best["fitness"]
        best_summary = {
            "generation": int(best.get("generation", -1)),
            "index": int(best.get("index", -1)),
            "name": ir.name,
            "theoretical_basis": getattr(ir, "theoretical_basis", ""),
            "operators_used": list(ir.operators_used),
            "hyperparams": dict(ir.hyperparams),
            "hf_like_score": float(fitness.get("hf_like_score", float("inf"))),
            "validation_objective": float(fitness.get("validation_objective", float("inf"))),
            "generalization_penalty": float(fitness.get("generalization_penalty", 0.0)),
            "pair_count": int(fitness.get("pair_count", 0) or 0),
        }

    summary = {
        "generations": int(generations),
        "population_size": int(population_size),
        "baseline_hf_score": float(baseline_hf_score) if baseline_hf_score is not None else None,
        "gate_failure_stats": {
            "by_code": failures_by_code,
        },
        "best_candidate": best_summary,
    }

    path = os.path.join(run_dir, "analysis_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


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
    float | None,
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
        baseline_early_valid,
        early_eval_steps,
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

    fitness = evaluate_free_loss_candidate(
        compiled,
        free_cfg,
        baseline_early_valid=baseline_early_valid,
        early_eval_steps=early_eval_steps,
    )

    # Proactively release unused CUDA cache in this worker process to
    # reduce fragmentation and long-lived reservations across jobs.
    if torch.cuda.is_available():
        try:
            device_str = str(hf_cfg.device)
            if device_str.startswith("cuda"):
                torch.cuda.set_device(torch.device(device_str))
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            # Cache cleanup is best-effort; ignore failures to avoid masking
            # the actual training result.
            pass

    return idx, fitness


def _device_worker(
    jobs: List[Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        str,
        List[str],
        str,
        int,
        int,
        float | None,
        int,
    ]],
    result_queue: "mp.Queue[Tuple[int, Dict[str, Any]]]",
) -> None:
    """Worker bound to a single device that evaluates its assigned jobs sequentially."""

    if not jobs:
        return

    # Best-effort debug logging: print which device this worker is bound to
    # and which candidates it will evaluate.
    try:
        import os  # local import to keep worker self-contained

        first_job = jobs[0]
        device_str = str(first_job[3])
        job_tags = [(int(j[6]), int(j[7])) for j in jobs]  # (generation, index)
        print(
            f"[free_loss_eoh][device_worker] pid={os.getpid()} device={device_str} "
            f"jobs={job_tags}",
            flush=True,
        )
    except Exception:  # noqa: BLE001
        pass

    for job in jobs:
        idx, fitness = _worker_evaluate_candidate(job)
        result_queue.put((idx, fitness))


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
    """Short-run HF-style evaluation using the original POMO po_loss.

    In addition to the final evaluation, this function records an
    intermediate validation objective after a small number of steps
    (early_eval_steps) so that candidate losses can be compared against
    the baseline at the same training horizon.
    """

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
    total_steps = get_total_hf_train_steps(cfg)
    # For early comparison we fix an absolute step budget (e.g., 100).
    early_eval_steps = min(100, total_steps)
    early_validation_objective: float | None = None

    LOGGER.info(
          "Baseline PO training: steps=%d, train_problem_size=%d, pomo_size=%d, "
          "batch_size=%d, device=%s, init_time=%.3fs",
        total_steps,
        cfg.train_problem_size,
        cfg.pomo_size,
        cfg.train_batch_size,
        str(device),
        t_init_end - t_init_start,
    )

    log_interval = max(total_steps // 20, 1)

    t_train_start = time.perf_counter()
    for step in range(total_steps):
        score, loss = _train_one_batch_with_po(env, model, optimizer, cfg)
        score_meter.update(score)
        loss_meter.update(loss)

        if (step + 1) % log_interval == 0 or step == 0:
            LOGGER.info(
                "Baseline PO step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f)",
                step + 1,
                total_steps,
                score,
                float(score_meter.avg),
                loss,
                float(loss_meter.avg),
            )
        if (step + 1) == early_eval_steps:
            # Early baseline performance for comparison with candidate losses.
            early_validation_objective = _evaluate_tsp_model(
                model=model,
                problem_size=cfg.train_problem_size,
                pomo_size=cfg.pomo_size,
                device=device,
                num_episodes=cfg.num_validation_episodes,
                batch_size=cfg.validation_batch_size,
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
        "early_validation_objective": early_validation_objective,
        "early_eval_steps": early_eval_steps,
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

    hf_epochs = int(cfg_yaml.get("hf_epochs", 0) or 0)
    hf_instances_per_epoch = int(cfg_yaml.get("hf_instances_per_epoch", 0) or 0)

    hf_cfg = HighFidelityConfig(
        problem=cfg_yaml.get("problem", "tsp"),
        hf_steps=int(cfg_yaml.get("f1_steps", 32)),
        hf_epochs=hf_epochs,
        hf_instances_per_epoch=hf_instances_per_epoch,
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
    baseline_early_valid: float | None = None
    burn_in_objectives: List[Dict[str, Any]] = []

    # Baseline: evaluate the original POMO po_loss once, using the same HF
    # configuration. This provides a reference score before searching over
    # free-form preference losses.
    try:
        baseline = evaluate_po_baseline(hf_cfg)
        baseline_hf_score = float(baseline["hf_score"])
        baseline_early_valid = float(
            baseline.get("early_validation_objective", baseline_hf_score)
        )
        burn_in_objectives.append(
            {
                "name": "po_loss_baseline",
                "type": "handcrafted_loss",
                "description": "Original POMO policy optimization loss (po_loss).",
                "hf_like_score": float(baseline["hf_score"]),
                "validation_objective": float(baseline["validation_objective"]),
                "generalization_penalty": float(baseline["generalization_penalty"]),
                "early_validation_objective": baseline.get("early_validation_objective"),
                "early_eval_steps": baseline.get("early_eval_steps"),
            }
        )
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
    repair_prompt = prompts.get("repair")
    expects_repair_prompt = prompts.get("expects_repair")

    out_root = cfg_yaml.get("output_root", "runs/free_loss_discovery")
    run_dir = _timestamp_dir(out_root)
    LOGGER.info("Run directory: %s", os.path.abspath(run_dir))

    candidates_log: List[Dict[str, Any]] = []
    gates_log: List[Dict[str, Any]] = []
    fitness_log: List[Dict[str, Any]] = []

    elites: List[Dict[str, Any]] = []

    max_repair_rounds = int(cfg_yaml.get("max_repair_rounds", 0) or 0)

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
        global_feedback = _build_global_feedback(
            elites=elites,
            gates_log=gates_log,
            burn_in_objectives=burn_in_objectives,
            baseline_hf_score=baseline_hf_score,
        )
        population: List[FreeLossIR] = []
        if gen == 0:
            LOGGER.info("Generating initial population with %d LLM candidates", init_llm)
            for _ in range(init_llm):
                ir = generate_free_loss_candidate(
                    gen_prompt,
                    operator_whitelist=operator_whitelist,
                    global_feedback=global_feedback,
                )
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
                    ir = crossover_free_loss(
                        crossover_prompt,
                        parent_irs[:2],
                        parents_fitness=[p["fitness"] for p in parents[:2]],
                        global_feedback=global_feedback,
                    )
                elif parent_irs and mutation_prompt:
                    ir = mutate_free_loss(
                        mutation_prompt,
                        parent_irs[0],
                        parent_fitness=parents[0]["fitness"],
                        global_feedback=global_feedback,
                    )
                else:
                    ir = generate_free_loss_candidate(
                        gen_prompt,
                        operator_whitelist=operator_whitelist,
                        global_feedback=global_feedback,
                    )
                population.append(_maybe_repair_expects(ir))

        LOGGER.info("Population size for generation %d: %d", gen, len(population))
        gen_elites: List[Dict[str, Any]] = []

        static_fail = 0
        dynamic_fail = 0
        evaluated = 0

        # Collect candidates that pass all gates and evaluate them in
        # parallel across available devices.
        eval_jobs: List[
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, List[str], str, int, int, float | None, int]
        ] = []
        eval_candidates: Dict[int, FreeLossIR] = {}

        for idx, original_ir in enumerate(population):
            ir = original_ir

            for attempt in range(max_repair_rounds + 1):
                static_res: StaticGateResult
                static_res = run_static_gates(ir, operator_whitelist=operator_whitelist)
                static_code = "" if static_res.ok else _classify_failure("static", static_res.reason)
                gate_entry: Dict[str, Any] = {
                    "generation": gen,
                    "index": idx,
                    "attempt": attempt,
                    "ir": asdict(ir),
                    "static_ok": static_res.ok,
                    "static_reason": static_res.reason,
                    "static_error_code": static_code,
                }

                if not static_res.ok:
                    if attempt < max_repair_rounds and repair_prompt:
                        try:
                            failure_payload = _build_failure_payload(
                                generation=gen,
                                index=idx,
                                attempt=attempt,
                                stage="static_gate",
                                code=static_code,
                                message=static_res.reason,
                                extra=None,
                            )
                            ir = repair_free_loss(repair_prompt, ir, failure_payload)
                            continue
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.warning(
                                "Failed to repair static gate error for gen=%d, idx=%d: %s",
                                gen,
                                idx,
                                exc,
                            )
                    static_fail += 1
                    gates_log.append(gate_entry)
                    break

                try:
                    compiled = compile_free_loss_candidate(ir, operator_whitelist=operator_whitelist)
                except CompileError as exc:
                    compile_code = _classify_failure("compile", str(exc))
                    gate_entry["dynamic_ok"] = False
                    gate_entry["dynamic_reason"] = f"compile_error: {exc}"
                    gate_entry["dynamic_error_code"] = compile_code
                    if attempt < max_repair_rounds and repair_prompt:
                        try:
                            failure_payload = _build_failure_payload(
                                generation=gen,
                                index=idx,
                                attempt=attempt,
                                stage="compile",
                                code=compile_code,
                                message=str(exc),
                                extra=None,
                            )
                            ir = repair_free_loss(repair_prompt, ir, failure_payload)
                            continue
                        except Exception as exc2:  # noqa: BLE001
                            LOGGER.warning(
                                "Failed to repair compile error for gen=%d, idx=%d: %s",
                                gen,
                                idx,
                                exc2,
                            )
                    gates_log.append(gate_entry)
                    break

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
                if not dyn_res.ok:
                    gate_entry["dynamic_error_code"] = _classify_failure("dynamic", dyn_res.reason)
                gates_log.append(gate_entry)

                if not dyn_res.ok:
                    if attempt < max_repair_rounds and repair_prompt:
                        LOGGER.warning(
                            "Dynamic gates failed for gen=%d, idx=%d, name=%s: reason=%s, "
                            "loss_value=%s, grad_norm=%s; attempting repair (attempt=%d/%d)",
                            gen,
                            idx,
                            ir.name,
                            dyn_res.reason,
                            str(dyn_res.loss_value),
                            str(dyn_res.grad_norm),
                            attempt + 1,
                            max_repair_rounds,
                        )
                        try:
                            failure_payload = _build_failure_payload(
                                generation=gen,
                                index=idx,
                                attempt=attempt,
                                stage="dynamic_gate",
                                code=gate_entry.get("dynamic_error_code", "E_DYNAMIC_OTHER"),
                                message=dyn_res.reason,
                                extra={
                                    "loss_value": dyn_res.loss_value,
                                    "grad_norm": dyn_res.grad_norm,
                                },
                            )
                            ir = repair_free_loss(repair_prompt, ir, failure_payload)
                            continue
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.warning(
                                "Failed to repair dynamic gate error for gen=%d, idx=%d: %s",
                                gen,
                                idx,
                                exc,
                            )
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
                    break

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
                        baseline_early_valid,
                        100,
                    )
                )
                break

        # Evaluate all surviving candidates in parallel across GPUs / devices.
        # We ensure that at any moment, at most one long-lived worker process
        # is active per device, so each GPU holds at most one candidate's
        # training state/cache.
        results: List[Tuple[int, Dict[str, Any]]] = []
        if eval_jobs:
            devices = _get_available_devices(hf_cfg.device)
            if not devices:
                devices = [hf_cfg.device]

            ctx = mp.get_context("spawn")

            # Partition jobs by device in a round-robin fashion.
            jobs_by_device: Dict[str, List[
                Tuple[
                    Dict[str, Any],
                    Dict[str, Any],
                    Dict[str, Any],
                    str,
                    List[str],
                    str,
                    int,
                    int,
                    float | None,
                    int,
                ]
            ]] = {dev: [] for dev in devices}

            for j_idx, job in enumerate(eval_jobs):
                dev = devices[j_idx % len(devices)]
                job_with_dev = list(job)
                job_with_dev[3] = dev
                jobs_by_device[dev].append(tuple(job_with_dev))  # type: ignore[arg-type]

            result_queue: "mp.Queue[Tuple[int, Dict[str, Any]]]" = ctx.Queue()
            processes: List[mp.Process] = []

            total_jobs = 0
            for dev, dev_jobs in jobs_by_device.items():
                if not dev_jobs:
                    continue
                total_jobs += len(dev_jobs)
                p = ctx.Process(target=_device_worker, args=(dev_jobs, result_queue))
                p.start()
                processes.append(p)

            # Collect all results.
            for _ in range(total_jobs):
                idx, fitness = result_queue.get()
                results.append((idx, fitness))

            for p in processes:
                p.join()

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
    _write_run_analysis(
        run_dir,
        baseline_hf_score=baseline_hf_score,
        generations=generations,
        population_size=population_size,
        gates_log=gates_log,
        elites=elites,
    )

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
