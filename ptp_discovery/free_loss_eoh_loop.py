from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import time
import logging
import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

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
    get_hf_epoch_plan,
    get_total_hf_train_steps,
)
from ptp_discovery.free_loss_compiler import CompileError
from ptp_discovery.free_loss_gates import (
    DynamicGateResult,
    PreferenceSemanticGateResult,
    StaticGateResult,
    run_dynamic_gates,
    run_preference_semantic_gates,
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
_REQUIRED_BATCH_KEYS = ("cost_a", "cost_b", "log_prob_w", "log_prob_l")


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
        if "duplicate_candidate" in msg:
            return "E_DUPLICATE"
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
        if "missing_dependency" in msg:
            return "E_MISSING_DEPENDENCY"
        if (
            "missing_batch_key" in msg
            or "extra_batch_key" in msg
            or "invalid_expects" in msg
            or "missing_expects" in msg
            or "unsupported_expects" in msg
        ):
            return "E_INPUT_MISMATCH"
        if "pref_semantic_violation" in msg:
            return "E_PREF_SEMANTIC"
        if "pref_" in msg:
            return "E_PREF_SEMANTIC"
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


class _SignatureNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # type: ignore[override]
        if isinstance(node.value, (int, float)):
            return ast.copy_location(ast.Constant(value=0), node)
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=""), node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:  # type: ignore[override]
        return ast.copy_location(ast.Name(id="v", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:  # type: ignore[override]
        return ast.copy_location(ast.arg(arg="v", annotation=None), node)


def _candidate_signature(ir: FreeLossIR) -> str:
    code_str = (ir.code or "").strip()
    if code_str:
        try:
            tree = ast.parse(code_str, mode="exec")
            tree = _SignatureNormalizer().visit(tree)
            ast.fix_missing_locations(tree)
            dump = ast.dump(tree, include_attributes=False)
        except Exception:
            dump = code_str
        digest = hashlib.sha1(dump.encode("utf-8")).hexdigest()
        return f"code:{digest}"

    ops = ",".join(sorted(ir.operators_used or []))
    hp_keys = ",".join(sorted((ir.hyperparams or {}).keys()))
    return f"template:{ops}|{hp_keys}"


def _behavior_descriptor(
    compiled: Any,
    *,
    deltas: Sequence[float],
    batch_size: int,
) -> List[float] | None:
    if not deltas:
        return []

    vec: List[float] = []
    for delta in deltas:
        cost_a = torch.rand(batch_size)
        gap = torch.rand(batch_size)
        cost_b = cost_a + gap

        log_prob_l = torch.empty(batch_size).uniform_(-20.0, 0.0)
        log_prob_w = log_prob_l + float(delta)
        log_prob_l = log_prob_l.clone().detach().requires_grad_(True)
        log_prob_w = log_prob_w.clone().detach().requires_grad_(True)

        batch = {
            "cost_a": cost_a,
            "cost_b": cost_b,
            "log_prob_w": log_prob_w,
            "log_prob_l": log_prob_l,
        }
        try:
            loss = compiled.loss_fn(batch=batch, model_output={}, extra={})
        except Exception:
            return None
        if not torch.isfinite(loss):
            return None

        grad_w, grad_l = torch.autograd.grad(
            loss,
            [log_prob_w, log_prob_l],
            allow_unused=True,
        )
        if grad_w is None or grad_l is None:
            return None

        vec.append(float(loss.item()))
        grad_delta = (grad_w - grad_l).mean().item()
        vec.append(float(grad_delta))

    return vec


def _jaccard_distance(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    inter = set_a & set_b
    return 1.0 - (len(inter) / len(union))


def _descriptor_distance(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    ops_weight: float,
    hparam_weight: float,
    behavior_weight: float,
) -> float:
    beh_a = a.get("behavior") or []
    beh_b = b.get("behavior") or []
    beh_dist = 0.0
    if beh_a and beh_b and len(beh_a) == len(beh_b):
        beh_dist = sum((x - y) ** 2 for x, y in zip(beh_a, beh_b)) ** 0.5

    ops_dist = _jaccard_distance(a.get("ops", []), b.get("ops", []))
    hp_dist = _jaccard_distance(a.get("hyperparams", []), b.get("hyperparams", []))

    return behavior_weight * beh_dist + ops_weight * ops_dist + hparam_weight * hp_dist


def _novelty_score(
    desc: Dict[str, Any] | None,
    archive: List[Dict[str, Any]],
    *,
    k: int,
    ops_weight: float,
    hparam_weight: float,
    behavior_weight: float,
) -> float:
    if desc is None:
        return 0.0
    if not archive:
        return 0.0
    distances = [
        _descriptor_distance(
            desc,
            other,
            ops_weight=ops_weight,
            hparam_weight=hparam_weight,
            behavior_weight=behavior_weight,
        )
        for other in archive
    ]
    distances.sort()
    top_k = distances[: max(1, min(k, len(distances)))]
    return float(sum(top_k) / len(top_k))


def _pareto_ranks(entries: List[Dict[str, Any]]) -> List[int]:
    n = len(entries)
    ranks = [0] * n
    dominated_by = [0] * n
    dominates: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        a_fit = float(a["fitness"]["hf_like_score"])
        b_fit = float(b["fitness"]["hf_like_score"])
        a_nov = float(a.get("novelty", 0.0))
        b_nov = float(b.get("novelty", 0.0))
        return (a_fit <= b_fit and a_nov >= b_nov) and (a_fit < b_fit or a_nov > b_nov)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(entries[i], entries[j]):
                dominates[i].append(j)
            elif _dominates(entries[j], entries[i]):
                dominated_by[i] += 1
        if dominated_by[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: List[int] = []
        for i in fronts[current]:
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    ranks[j] = current + 1
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    return ranks


def _select_parents(entries: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not entries:
        return []
    ranks = _pareto_ranks(entries)
    ranked = sorted(
        zip(entries, ranks),
        key=lambda x: (x[1], -float(x[0].get("novelty", 0.0)), float(x[0]["fitness"]["hf_like_score"])),
    )
    return [entry for entry, _ in ranked[:k]]


def _summarize_diversity(archive: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    op_counts: Dict[str, int] = {}
    hp_counts: Dict[str, int] = {}
    for entry in archive:
        ops = tuple(sorted(entry.get("ops", [])))
        op_counts[str(ops)] = op_counts.get(str(ops), 0) + 1
        hps = tuple(sorted(entry.get("hyperparams", [])))
        hp_counts[str(hps)] = hp_counts.get(str(hps), 0) + 1

    op_common = sorted(op_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    hp_common = sorted(hp_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    return {
        "common_operator_sets": [{"operators_used": k, "count": v} for k, v in op_common],
        "common_hparam_keys": [{"hyperparams": k, "count": v} for k, v in hp_common],
    }


def _build_auto_seed_objectives() -> List[Dict[str, Any]]:
    return [
        {
            "name": "seed_logsigmoid",
            "type": "auto_seed",
            "description": "Base Bradley-Terry style loss: -logsigmoid(alpha * (logp_w - logp_l)).",
            "operators_used": ["logsigmoid"],
            "hyperparams": {"alpha": 1.0},
        },
        {
            "name": "seed_softplus_margin",
            "type": "auto_seed",
            "description": "Softplus hinge: softplus(margin - (logp_w - logp_l)).",
            "operators_used": ["softplus"],
            "hyperparams": {"margin": 0.5},
        },
        {
            "name": "seed_focal_logsigmoid",
            "type": "auto_seed",
            "description": "Focal modulated: exp(gamma * log(1 - sigmoid(delta))) * -logsigmoid(delta).",
            "operators_used": ["sigmoid", "logsigmoid", "exp", "log"],
            "hyperparams": {"gamma": 2.0},
        },
        {
            "name": "seed_cost_gap_margin",
            "type": "auto_seed",
            "description": "Margin grows with cost gap: margin = beta * (cost_b - cost_a).",
            "operators_used": ["logsigmoid"],
            "hyperparams": {"beta": 1.0},
        },
        {
            "name": "seed_softplus_cost_scale",
            "type": "auto_seed",
            "description": "Scale by softplus(cost_gap): softplus(cost_gap) * -logsigmoid(delta).",
            "operators_used": ["softplus", "logsigmoid"],
            "hyperparams": {},
        },
        {
            "name": "seed_tanh_margin",
            "type": "auto_seed",
            "description": "Tanh margin: margin = tanh(scale * cost_gap).",
            "operators_used": ["tanh", "logsigmoid"],
            "hyperparams": {"scale": 1.0},
        },
        {
            "name": "seed_exp_penalty",
            "type": "auto_seed",
            "description": "Exponential penalty on negative delta: exp(-delta).",
            "operators_used": ["exp"],
            "hyperparams": {},
        },
        {
            "name": "seed_sigmoid_margin",
            "type": "auto_seed",
            "description": "Sigmoid margin: margin = sigmoid(beta * cost_gap).",
            "operators_used": ["sigmoid", "logsigmoid"],
            "hyperparams": {"beta": 1.0},
        },
        {
            "name": "seed_mix_logsigmoid_softplus",
            "type": "auto_seed",
            "description": "Mixture: alpha * logsigmoid + (1-alpha) * softplus hinge.",
            "operators_used": ["logsigmoid", "softplus", "sigmoid"],
            "hyperparams": {"alpha": 0.5},
        },
        {
            "name": "seed_zscore_margin",
            "type": "auto_seed",
            "description": "Z-score cost gap to set margin scale.",
            "operators_used": ["zscore", "logsigmoid"],
            "hyperparams": {"scale": 1.0},
        },
    ]

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
    diversity_archive: List[Dict[str, Any]] | None = None,
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
                "epoch_objective_mean": float(fitness.get("epoch_objective_mean", float("inf")))
                if fitness.get("epoch_objective_mean") is not None
                else None,
                "epoch_baseline_violations": fitness.get("epoch_baseline_violations"),
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

    diversity_summary = _summarize_diversity(diversity_archive or [])

    return {
        "burn_in_objectives": burn_in_objectives,
        "recent_elites": elite_summaries,
        "recent_failures": {
            "by_code": failures_by_code,
            "examples": failure_examples,
        },
        "diversity_summary": diversity_summary,
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
            "epoch_objective_mean": float(fitness.get("epoch_objective_mean", float("inf")))
            if fitness.get("epoch_objective_mean") is not None
            else None,
            "epoch_baseline_violations": fitness.get("epoch_baseline_violations"),
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


def _compute_early_eval_steps(cfg_yaml: Dict[str, Any], hf_cfg: HighFidelityConfig) -> int:
    total_steps = get_total_hf_train_steps(hf_cfg)
    early_eval_epochs = int(cfg_yaml.get("early_eval_epochs", 0) or 0)
    early_eval_instances_per_epoch = int(
        cfg_yaml.get("early_eval_instances_per_epoch", 0) or 0
    )
    early_eval_steps_cfg = cfg_yaml.get("early_eval_steps")

    if early_eval_epochs > 0:
        instances_per_epoch = early_eval_instances_per_epoch
        if instances_per_epoch <= 0:
            instances_per_epoch = int(hf_cfg.hf_instances_per_epoch or 0)
        if instances_per_epoch > 0:
            batch_size = max(int(hf_cfg.train_batch_size), 1)
            steps_per_epoch = math.ceil(instances_per_epoch / batch_size)
            steps = early_eval_epochs * steps_per_epoch
        else:
            steps = 0
    elif early_eval_steps_cfg is not None:
        steps = int(early_eval_steps_cfg or 0)
    else:
        steps = min(100, total_steps)

    if steps <= 0:
        return 0
    return min(int(steps), int(total_steps))


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
    List[float] | None,
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
        baseline_epoch_objectives,
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
        baseline_epoch_violation_weight=float(
            free_cfg_dict.get("baseline_epoch_violation_weight", 1.0)
        ),
    )

    # Reconstruct IR and compiled loss in the worker.
    ir = ir_from_json(ir_payload)
    fitness: Dict[str, Any]
    worst_score = float(1e9)

    try:
        compiled = compile_free_loss_candidate(ir, operator_whitelist=operator_whitelist)
        fitness = evaluate_free_loss_candidate(
            compiled,
            free_cfg,
            baseline_early_valid=baseline_early_valid,
            baseline_epoch_objectives=baseline_epoch_objectives,
            early_eval_steps=early_eval_steps,
        )
    except Exception as exc:  # noqa: BLE001
        # If candidate evaluation fails (e.g., NaNs in probabilities or loss),
        # treat this candidate as having the worst possible fitness instead of
        # crashing the worker process.
        try:
            fl_logger = logging.getLogger("fitness.free_loss_fidelity")
            fl_logger.exception(
                "Candidate evaluation failed for gen=%d, idx=%d; assigning worst fitness.",
                gen,
                idx,
            )
        except Exception:  # noqa: BLE001
            pass

        fitness = {
            "hf_like_score": worst_score,
            "validation_objective": worst_score,
            "generalization_penalty": 0.0,
            "generalization_objectives": {},
            "epoch_objective_mean": None,
            "epoch_baseline_violations": None,
            "epoch_better_than_baseline": None,
            "train_score_mean": float("nan"),
            "train_loss_mean": float("nan"),
            "pair_count": 0,
            "early_eval": {
                "enabled": False,
                "steps": int(early_eval_steps or 0),
                "baseline_validation_objective": baseline_early_valid,
                "candidate_validation_objective": None,
                "early_stopped": False,
            },
            "epoch_eval": {
                "enabled": False,
                "steps_per_epoch": None,
                "epochs_total": 0,
                "objectives": [],
                "objective_mean": None,
                "baseline_margins": None,
                "baseline_violations": None,
                "better_than_baseline": None,
            },
        }
        fitness["eval_error"] = str(exc)
    finally:
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
        List[float] | None,
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


def evaluate_po_baseline(
    cfg: HighFidelityConfig,
    *,
    early_eval_steps: int | None = None,
) -> Dict[str, Any]:
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
    steps_per_epoch, epochs_total = get_hf_epoch_plan(cfg)
    epoch_validation_objectives: List[float] = []
    if early_eval_steps is None:
        early_eval_steps = min(100, total_steps)
    else:
        early_eval_steps = min(max(int(early_eval_steps), 0), total_steps)
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
        if steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0:
            epoch_idx = (step + 1) // steps_per_epoch
            epoch_valid_obj = _evaluate_tsp_model(
                model=model,
                problem_size=cfg.train_problem_size,
                pomo_size=cfg.pomo_size,
                device=device,
                num_episodes=cfg.num_validation_episodes,
                batch_size=cfg.validation_batch_size,
            )
            epoch_validation_objectives.append(epoch_valid_obj)
            LOGGER.info(
                "Baseline PO epoch %d/%d: validation_objective=%.6f",
                epoch_idx,
                epochs_total,
                epoch_valid_obj,
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

    epoch_objective_mean: float | None = None
    if epoch_validation_objectives:
        epoch_objective_mean = float(
            sum(epoch_validation_objectives) / len(epoch_validation_objectives)
        )

    base_objective = (
        epoch_objective_mean if epoch_objective_mean is not None else main_valid_obj
    )
    hf_score = main_valid_obj + cfg.generalization_penalty_weight * generalization_penalty
    fitness_score = base_objective + cfg.generalization_penalty_weight * generalization_penalty

    LOGGER.info(
        "Baseline PO timing: init=%.3fs, train=%.3fs, eval=%.3fs",
        t_init_end - t_init_start,
        t_train_end - t_train_start,
        t_eval_end - t_eval_start,
    )

    return {
        "hf_score": hf_score,
        "fitness_score": fitness_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "early_validation_objective": early_validation_objective,
        "early_eval_steps": early_eval_steps,
        "epoch_eval": {
            "enabled": bool(steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch > 0 else None,
            "epochs_total": int(epochs_total),
            "objectives": epoch_validation_objectives,
            "objective_mean": epoch_objective_mean,
        },
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
        baseline_epoch_violation_weight=float(
            cfg_yaml.get("baseline_epoch_violation_weight", 1.0)
        ),
    )
    hf_cfg_dict: Dict[str, Any] = dict(hf_cfg.__dict__)
    free_cfg_dict: Dict[str, Any] = {
        "f1_steps": free_cfg.f1_steps,
        "f2_steps": free_cfg.f2_steps,
        "f3_enabled": free_cfg.f3_enabled,
        "baseline_epoch_violation_weight": free_cfg.baseline_epoch_violation_weight,
    }

    early_eval_steps = _compute_early_eval_steps(cfg_yaml, hf_cfg)
    baseline_hf_score: float | None = None
    baseline_early_valid: float | None = None
    baseline_epoch_objectives: List[float] | None = None
    burn_in_objectives: List[Dict[str, Any]] = []
    diversity_archive: List[Dict[str, Any]] = []
    diverse_elites: List[Dict[str, Any]] = []
    seen_signatures: set[str] = set()

    # Baseline: evaluate the original POMO po_loss once, using the same HF
    # configuration. This provides a reference score before searching over
    # free-form preference losses.
    try:
        baseline = evaluate_po_baseline(hf_cfg, early_eval_steps=early_eval_steps)
        baseline_hf_score = float(baseline.get("fitness_score", baseline["hf_score"]))
        baseline_early_valid = float(
            baseline.get("early_validation_objective", baseline["validation_objective"])
        )
        baseline_epoch_objectives = baseline.get("epoch_eval", {}).get("objectives")
        if baseline_epoch_objectives:
            baseline_epoch_objectives = [float(v) for v in baseline_epoch_objectives]
        burn_in_objectives.append(
            {
                "name": "po_loss_baseline",
                "type": "handcrafted_loss",
                "description": "Original POMO policy optimization loss (po_loss).",
                "hf_like_score": float(baseline.get("fitness_score", baseline["hf_score"])),
                "fitness_score": float(baseline.get("fitness_score", baseline["hf_score"])),
                "validation_objective": float(baseline["validation_objective"]),
                "generalization_penalty": float(baseline["generalization_penalty"]),
                "early_validation_objective": baseline.get("early_validation_objective"),
                "early_eval_steps": baseline.get("early_eval_steps"),
                "epoch_objective_mean": baseline.get("epoch_eval", {}).get("objective_mean"),
                "epoch_validation_objectives": baseline.get("epoch_eval", {}).get("objectives"),
                "epoch_steps_per_epoch": baseline.get("epoch_eval", {}).get("steps_per_epoch"),
            }
        )
        LOGGER.info(
            "Baseline po_loss: hf_score=%.6f, fitness_score=%.6f, validation_objective=%.6f, gen_penalty=%.6f",
            baseline["hf_score"],
            baseline.get("fitness_score", baseline["hf_score"]),
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
    max_resample_rounds = int(cfg_yaml.get("max_resample_rounds", 1) or 0)
    burn_in_objectives_auto = bool(cfg_yaml.get("burn_in_objectives_auto", True))

    pref_semantic_gate_enabled = bool(cfg_yaml.get("pref_semantic_gate_enabled", True))
    pref_semantic_trials = int(cfg_yaml.get("pref_semantic_trials", 6))
    pref_semantic_batch_size = int(cfg_yaml.get("pref_semantic_batch_size", 128))
    pref_semantic_min_pass_rate = float(cfg_yaml.get("pref_semantic_min_pass_rate", 0.8))
    pref_semantic_swap_tolerance = float(cfg_yaml.get("pref_semantic_swap_tolerance", 1e-3))
    pref_semantic_gap_min_ratio = float(cfg_yaml.get("pref_semantic_gap_min_ratio", 0.9))

    behavior_deltas = cfg_yaml.get("novelty_behavior_deltas") or [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    behavior_deltas = [float(v) for v in behavior_deltas]
    novelty_k = int(cfg_yaml.get("novelty_k", 5))
    novelty_ops_weight = float(cfg_yaml.get("novelty_ops_weight", 1.0))
    novelty_hparam_weight = float(cfg_yaml.get("novelty_hparam_weight", 0.5))
    novelty_behavior_weight = float(cfg_yaml.get("novelty_behavior_weight", 1.0))
    diversity_archive_size = int(cfg_yaml.get("diversity_archive_size", 32))

    out_root = cfg_yaml.get("output_root", "runs/free_loss_discovery")
    run_dir = _timestamp_dir(out_root)
    LOGGER.info("Run directory: %s", os.path.abspath(run_dir))

    candidates_log: List[Dict[str, Any]] = []
    gates_log: List[Dict[str, Any]] = []
    fitness_log: List[Dict[str, Any]] = []

    elites: List[Dict[str, Any]] = []

    max_repair_rounds = int(cfg_yaml.get("max_repair_rounds", 0) or 0)

    if burn_in_objectives_auto:
        burn_in_objectives.extend(_build_auto_seed_objectives())

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

    def _sample_candidate(
        *,
        parent_irs: List[FreeLossIR],
        parents: List[Dict[str, Any]],
        global_feedback: Dict[str, Any],
    ) -> FreeLossIR:
        if len(parent_irs) >= 2 and crossover_prompt:
            return crossover_free_loss(
                crossover_prompt,
                parent_irs[:2],
                parents_fitness=[p["fitness"] for p in parents[:2]],
                global_feedback=global_feedback,
            )
        if parent_irs and mutation_prompt:
            return mutate_free_loss(
                mutation_prompt,
                parent_irs[0],
                parent_fitness=parents[0]["fitness"],
                global_feedback=global_feedback,
            )
        return generate_free_loss_candidate(
            gen_prompt,
            operator_whitelist=operator_whitelist,
            global_feedback=global_feedback,
        )

    def _should_resample_dynamic(reason: str) -> bool:
        msg = (reason or "").lower()
        return (
            "missing_dependency" in msg
            or "missing_batch_key" in msg
            or "extra_batch_key" in msg
            or "invalid_expects" in msg
            or "missing_expects" in msg
            or "unsupported_expects" in msg
        )

    for gen in range(generations):
        LOGGER.info("=== Generation %d/%d ===", gen, generations - 1)
        global_feedback = _build_global_feedback(
            elites=elites,
            gates_log=gates_log,
            burn_in_objectives=burn_in_objectives,
            baseline_hf_score=baseline_hf_score,
            diversity_archive=diversity_archive,
        )
        population: List[FreeLossIR] = []
        parents: List[Dict[str, Any]] = []
        parent_irs: List[FreeLossIR] = []
        if gen == 0:
            LOGGER.info("Generating initial population with %d LLM candidates", init_llm)
            for _ in range(init_llm):
                ir = _sample_candidate(
                    parent_irs=parent_irs,
                    parents=parents,
                    global_feedback=global_feedback,
                )
                population.append(_maybe_repair_expects(ir))
        else:
            parent_pool = diverse_elites if diverse_elites else elites
            parents = _select_parents(parent_pool, elite_size)
            parent_irs = [p["ir"] for p in parents]
            LOGGER.info(
                "Generating population via crossover/mutation: size=%d, elite_size=%d, available_elites=%d",
                population_size,
                elite_size,
                len(parent_irs),
            )
            for _ in range(population_size):
                ir = _sample_candidate(
                    parent_irs=parent_irs,
                    parents=parents,
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
                List[float] | None,
                int,
            ]
        ] = []
        eval_candidates: Dict[int, FreeLossIR] = {}

        for idx, original_ir in enumerate(population):
            ir = original_ir
            resample_attempts = 0
            while True:
                resampled = False
                for attempt in range(max_repair_rounds + 1):
                    signature = _candidate_signature(ir)
                    static_res: StaticGateResult
                    if signature in seen_signatures:
                        static_res = StaticGateResult(ok=False, reason="duplicate_candidate")
                    else:
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
                        compiled = compile_free_loss_candidate(
                            ir,
                            operator_whitelist=operator_whitelist,
                        )
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
                    }

                    dyn_res: DynamicGateResult
                    dyn_res = run_dynamic_gates(
                        compiled,
                        batch=dummy_batch,
                        model=model,
                        required_batch_keys=_REQUIRED_BATCH_KEYS,
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

                    if not dyn_res.ok:
                        gates_log.append(gate_entry)
                        if _should_resample_dynamic(dyn_res.reason) and resample_attempts < max_resample_rounds:
                            resample_attempts += 1
                            LOGGER.warning(
                                "Dynamic gates failed for gen=%d, idx=%d, name=%s: reason=%s; "
                                "dropping candidate and resampling (attempt=%d/%d)",
                                gen,
                                idx,
                                ir.name,
                                dyn_res.reason,
                                resample_attempts,
                                max_resample_rounds,
                            )
                            ir = _maybe_repair_expects(
                                _sample_candidate(
                                    parent_irs=parent_irs,
                                    parents=parents,
                                    global_feedback=global_feedback,
                                )
                            )
                            resampled = True
                            break
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

                    pref_res: PreferenceSemanticGateResult | None = None
                    if pref_semantic_gate_enabled:
                        pref_res = run_preference_semantic_gates(
                            compiled,
                            trials=pref_semantic_trials,
                            batch_size=pref_semantic_batch_size,
                            min_pass_rate=pref_semantic_min_pass_rate,
                            swap_tolerance=pref_semantic_swap_tolerance,
                            gap_min_ratio=pref_semantic_gap_min_ratio,
                        )
                        gate_entry.update(
                            {
                                "pref_ok": pref_res.ok,
                                "pref_reason": pref_res.reason,
                                "pref_mono_pass_rate": pref_res.mono_pass_rate,
                                "pref_swap_pass_rate": pref_res.swap_pass_rate,
                                "pref_gap_pass_rate": pref_res.gap_pass_rate,
                            }
                        )
                        if not pref_res.ok:
                            gate_entry["dynamic_error_code"] = _classify_failure(
                                "dynamic",
                                pref_res.reason,
                            )
                            gates_log.append(gate_entry)
                            if attempt < max_repair_rounds and repair_prompt:
                                LOGGER.warning(
                                    "Preference gates failed for gen=%d, idx=%d, name=%s: reason=%s; "
                                    "attempting repair (attempt=%d/%d)",
                                    gen,
                                    idx,
                                    ir.name,
                                    pref_res.reason,
                                    attempt + 1,
                                    max_repair_rounds,
                                )
                                try:
                                    failure_payload = _build_failure_payload(
                                        generation=gen,
                                        index=idx,
                                        attempt=attempt,
                                        stage="preference_gate",
                                        code=gate_entry.get("dynamic_error_code", "E_PREF_SEMANTIC"),
                                        message=pref_res.reason,
                                        extra={
                                            "mono_pass_rate": pref_res.mono_pass_rate,
                                            "swap_pass_rate": pref_res.swap_pass_rate,
                                            "gap_pass_rate": pref_res.gap_pass_rate,
                                        },
                                    )
                                    ir = repair_free_loss(repair_prompt, ir, failure_payload)
                                    continue
                                except Exception as exc:  # noqa: BLE001
                                    LOGGER.warning(
                                        "Failed to repair preference gate error for gen=%d, idx=%d: %s",
                                        gen,
                                        idx,
                                        exc,
                                    )
                            dynamic_fail += 1
                            LOGGER.warning(
                                "Preference gates failed for gen=%d, idx=%d, name=%s: reason=%s",
                                gen,
                                idx,
                                ir.name,
                                pref_res.reason,
                            )
                            break

                    if pref_res is None or pref_res.ok:
                        gates_log.append(gate_entry)

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
                            baseline_epoch_objectives,
                            early_eval_steps,
                        )
                    )
                    seen_signatures.add(signature)
                    break
                if resampled:
                    continue
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
                    List[float] | None,
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

            descriptor: Dict[str, Any] | None = None
            novelty = 0.0
            try:
                compiled = compile_free_loss_candidate(ir, operator_whitelist=operator_whitelist)
                behavior = _behavior_descriptor(
                    compiled,
                    deltas=behavior_deltas,
                    batch_size=pref_semantic_batch_size,
                )
                if behavior is not None:
                    descriptor = {
                        "behavior": behavior,
                        "ops": list(ir.operators_used or []),
                        "hyperparams": list((ir.hyperparams or {}).keys()),
                        "signature": _candidate_signature(ir),
                    }
                    novelty = _novelty_score(
                        descriptor,
                        diversity_archive,
                        k=novelty_k,
                        ops_weight=novelty_ops_weight,
                        hparam_weight=novelty_hparam_weight,
                        behavior_weight=novelty_behavior_weight,
                    )
            except Exception:  # noqa: BLE001
                descriptor = None
                novelty = 0.0

            fitness["novelty"] = novelty

            hf_like_score = float(fitness["hf_like_score"])
            epoch_mean = fitness.get("epoch_objective_mean")
            epoch_mean_val = float(epoch_mean) if epoch_mean is not None else float("nan")
            epoch_violations = fitness.get("epoch_baseline_violations")
            better_than_baseline = None
            if baseline_epoch_objectives is not None:
                better_than_baseline = bool(epoch_violations == 0)
            elif baseline_hf_score is not None:
                # Lower score is better.
                better_than_baseline = hf_like_score <= baseline_hf_score

            LOGGER.info(
                "Gen %d cand %d: hf_like_score=%.6f, validation_objective=%.6f, "
                "epoch_mean=%.6f, epoch_violations=%s, baseline=%.6f, better_than_baseline=%s",
                gen,
                idx,
                hf_like_score,
                float(fitness["validation_objective"]),
                epoch_mean_val,
                str(epoch_violations),
                float(baseline_hf_score) if baseline_hf_score is not None else float("nan"),
                str(better_than_baseline),
            )

            cand_entry_log = {
                "generation": gen,
                "index": idx,
                "ir": asdict(ir),
                "fitness": fitness,
                "better_than_baseline": better_than_baseline,
                "novelty": novelty,
                "diversity_descriptor": descriptor,
            }
            candidates_log.append(cand_entry_log)
            fitness_log.append(
                {
                    "generation": gen,
                    "index": idx,
                    "hf_like_score": fitness["hf_like_score"],
                    "validation_objective": fitness["validation_objective"],
                    "epoch_objective_mean": epoch_mean,
                    "epoch_baseline_violations": epoch_violations,
                    "epoch_better_than_baseline": fitness.get("epoch_better_than_baseline"),
                    "baseline_hf_score": baseline_hf_score,
                    "better_than_baseline": better_than_baseline,
                    "novelty": novelty,
                }
            )
            elite_entry = {
                "generation": gen,
                "index": idx,
                "ir": ir,
                "fitness": fitness,
                "novelty": novelty,
            }
            gen_elites.append(elite_entry)

            if descriptor is not None:
                diversity_archive.append(descriptor)
                if len(diversity_archive) > diversity_archive_size:
                    diversity_archive.pop(0)

        def _elite_key(entry: Dict[str, Any]) -> Tuple[float, float, float]:
            score = float(entry["fitness"]["hf_like_score"])
            pair_count = float(entry["fitness"].get("pair_count", 0) or 0)
            violations = entry["fitness"].get("epoch_baseline_violations")
            if violations is not None:
                return (float(violations), score, pair_count)
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

        if gen_elites:
            diverse_pool = diverse_elites + gen_elites
            diverse_elites = _select_parents(
                diverse_pool,
                min(diversity_archive_size, len(diverse_pool)),
            )

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
