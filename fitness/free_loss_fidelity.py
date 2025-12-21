from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import logging
import torch
from torch.optim import Adam

from .ptp_high_fidelity import (
    HighFidelityConfig,
    _set_seed,
    _evaluate_tsp_model,
    get_hf_epoch_plan,
    get_total_hf_train_steps,
)
from ptp_discovery.free_loss_compiler import CompiledFreeLoss

from TSPEnv import TSPEnv  # type: ignore  # noqa: E402
from TSPModel import TSPModel  # type: ignore  # noqa: E402
from utils.utils import AverageMeter  # type: ignore  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass
class FreeLossFidelityConfig:
    hf: HighFidelityConfig
    f1_steps: int = 32
    f2_steps: int = 0
    f3_enabled: bool = False
    baseline_epoch_violation_weight: float = 1.0


def _build_preference_pairs(
    objective: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Vectorized construction of winner/loser indices.

    For each instance in the batch, we consider all pairs (i, j) such that
    objective[i] < objective[j] (i is better than j). This yields three
    index tensors (batch_idx, winner_idx, loser_idx) plus the total pair
    count. We intentionally do not compute structural features here to keep
    the free-loss evaluation lightweight.
    """

    # objective: (batch, pomo)
    # mask[b, i, j] = True if i is better (lower cost) than j for instance b.
    mask = objective[:, :, None] < objective[:, None, :]
    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
    pair_count = int(b_idx.numel())
    return (b_idx, winner_idx, loser_idx), pair_count


def _train_one_batch_with_free_loss(
    env: TSPEnv,
    model: TSPModel,
    optimizer: Adam,
    compiled_loss: CompiledFreeLoss,
    hf_cfg: HighFidelityConfig,
) -> Tuple[float, float, int]:
    batch_size = hf_cfg.train_batch_size
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

    objective = -reward  # (batch, pomo)
    log_prob = torch.log(prob_list + 1e-8).sum(dim=2)  # (batch, pomo)

    (b_idx, winner_idx, loser_idx), pair_count = _build_preference_pairs(objective)

    if pair_count == 0:
        # Fallback to a simple policy-gradient-style loss when no preference
        # pairs exist. This keeps training stable without imposing additional
        # theoretical structure beyond the candidate loss itself.
        advantage = reward - reward.mean(dim=1, keepdim=True)
        rl_log_prob = log_prob
        loss = -(advantage * rl_log_prob).mean()
    else:
        cost_a_tensor = objective[b_idx, winner_idx]
        cost_b_tensor = objective[b_idx, loser_idx]
        logp_w_tensor = log_prob[b_idx, winner_idx]
        logp_l_tensor = log_prob[b_idx, loser_idx]

        batch = {
            "cost_a": cost_a_tensor,
            "cost_b": cost_b_tensor,
            "log_prob_w": logp_w_tensor,
            "log_prob_l": logp_l_tensor,
        }
        loss = compiled_loss.loss_fn(
            batch=batch,
            model_output={},
            extra={"alpha": hf_cfg.alpha},
        )

    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return score_mean.item(), float(loss.item()), pair_count


def evaluate_free_loss_candidate(
    compiled_loss: CompiledFreeLoss,
    cfg: FreeLossFidelityConfig,
    *,
    baseline_early_valid: float | None = None,
    early_eval_steps: int = 0,
    baseline_epoch_objectives: Sequence[float] | None = None,
) -> Dict[str, Any]:
    _set_seed(cfg.hf.seed)

    device_str = cfg.hf.device
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

    import time as _time  # local alias to avoid confusion
    t_init_start = _time.perf_counter()

    env = TSPEnv(
        problem_size=cfg.hf.train_problem_size,
        pomo_size=cfg.hf.pomo_size,
        device=str(device),
    )

    model = TSPModel(**model_params).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(cfg.hf.learning_rate),
        weight_decay=float(cfg.hf.weight_decay),
    )
    t_init_end = _time.perf_counter()

    score_meter = AverageMeter()
    loss_meter = AverageMeter()
    total_pairs = 0

    # Multi-phase training: first run a short F1 phase determined by the
    # HF config (hf_steps / hf_epochs / hf_instances_per_epoch), then
    # optionally extend with an additional F2 budget specified in the
    # free-loss config. This keeps phase boundaries explicit in the
    # returned metrics while still sharing a single model.
    steps_f1 = get_total_hf_train_steps(cfg.hf)
    steps_f2 = max(int(cfg.f2_steps), 0)
    steps = steps_f1 + steps_f2
    steps_per_epoch, epochs_total = get_hf_epoch_plan(cfg.hf)
    epoch_validation_objectives: List[float] = []

    score_meter_f1 = AverageMeter()
    loss_meter_f1 = AverageMeter()
    total_pairs_f1 = 0

    logger.info(
        "Free-loss training: f1_steps=%d, f2_steps=%d, total_steps=%d, train_problem_size=%d, pomo_size=%d, batch_size=%d, device=%s",
        steps_f1,
        steps_f2,
        steps,
        cfg.hf.train_problem_size,
        cfg.hf.pomo_size,
        cfg.hf.train_batch_size,
        str(device),
    )

    log_interval = max(steps // 10, 1)

    t_train_start = _time.perf_counter()

    # Early-stop metadata relative to the baseline.
    early_eval_steps = max(int(early_eval_steps or 0), 0)
    early_eval_effective = min(early_eval_steps, steps) if early_eval_steps > 0 else 0
    early_validation_objective: float | None = None
    early_stopped = False

    for step in range(steps):
        score, loss, pair_count = _train_one_batch_with_free_loss(
            env=env,
            model=model,
            optimizer=optimizer,
            compiled_loss=compiled_loss,
            hf_cfg=cfg.hf,
        )
        score_meter.update(score)
        loss_meter.update(loss)
        total_pairs += int(pair_count)

        if step < steps_f1:
            score_meter_f1.update(score)
            loss_meter_f1.update(loss)
            total_pairs_f1 += int(pair_count)

        if (step + 1) % log_interval == 0 or step == 0:
            logger.info(
                "Free-loss step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f), pairs_step=%d, pairs_total=%d",
                step + 1,
                steps,
                score,
                float(score_meter.avg),
                loss,
                float(loss_meter.avg),
                int(pair_count),
                total_pairs,
            )

        if steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0:
            epoch_idx = (step + 1) // steps_per_epoch
            if epoch_idx <= epochs_total:
                epoch_valid_obj = _evaluate_tsp_model(
                    model=model,
                    problem_size=cfg.hf.train_problem_size,
                    pomo_size=cfg.hf.pomo_size,
                    device=device,
                    num_episodes=cfg.hf.num_validation_episodes,
                    batch_size=cfg.hf.validation_batch_size,
                )
                epoch_validation_objectives.append(epoch_valid_obj)
                logger.info(
                    "Free-loss epoch %d/%d: validation_objective=%.6f",
                    epoch_idx,
                    epochs_total,
                    epoch_valid_obj,
                )

        # Early evaluation for comparison with the baseline after a small
        # fixed number of steps (e.g., 100). If the candidate performs
        # strictly worse than the baseline at this horizon, we stop training
        # early to save compute.
        if early_eval_effective > 0 and (step + 1) == early_eval_effective:
            early_validation_objective = _evaluate_tsp_model(
                model=model,
                problem_size=cfg.hf.train_problem_size,
                pomo_size=cfg.hf.pomo_size,
                device=device,
                num_episodes=cfg.hf.num_validation_episodes,
                batch_size=cfg.hf.validation_batch_size,
            )
            if baseline_early_valid is not None and early_validation_objective > baseline_early_valid:
                early_stopped = True
                logger.info(
                    "Early stop at step %d: candidate early_valid=%.6f baseline_early=%.6f",
                    step + 1,
                    early_validation_objective,
                    baseline_early_valid,
                )
                break
    t_train_end = _time.perf_counter()

    t_eval_start = _time.perf_counter()
    if early_stopped and early_validation_objective is not None:
        main_valid_obj = float(early_validation_objective)
    else:
        main_valid_obj = _evaluate_tsp_model(
            model=model,
            problem_size=cfg.hf.train_problem_size,
            pomo_size=cfg.hf.pomo_size,
            device=device,
            num_episodes=cfg.hf.num_validation_episodes,
            batch_size=cfg.hf.validation_batch_size,
        )

    gen_objectives: Dict[int, float] = {}
    for size in cfg.hf.valid_problem_sizes:
        size_int = int(size)
        gen_obj = _evaluate_tsp_model(
            model=model,
            problem_size=size_int,
            pomo_size=cfg.hf.pomo_size,
            device=device,
            num_episodes=cfg.hf.num_validation_episodes,
            batch_size=cfg.hf.validation_batch_size,
        )
        gen_objectives[size_int] = gen_obj
    t_eval_end = _time.perf_counter()

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    epoch_objective_mean: float | None = None
    if epoch_validation_objectives:
        epoch_objective_mean = float(
            sum(epoch_validation_objectives) / len(epoch_validation_objectives)
        )

    epoch_baseline_violations: int | None = None
    epoch_better_than_baseline: bool | None = None
    epoch_baseline_margins: List[float] | None = None
    if baseline_epoch_objectives:
        baseline_list = [float(v) for v in baseline_epoch_objectives]
        compare_len = min(len(epoch_validation_objectives), len(baseline_list))
        epoch_baseline_margins = []
        for i in range(compare_len):
            margin = float(epoch_validation_objectives[i]) - baseline_list[i]
            epoch_baseline_margins.append(margin)
        violations = sum(1 for m in epoch_baseline_margins if m > 0.0)
        missing = max(len(baseline_list) - len(epoch_validation_objectives), 0)
        violations += missing
        epoch_baseline_violations = int(violations)
        epoch_better_than_baseline = epoch_baseline_violations == 0

    base_objective = (
        epoch_objective_mean if epoch_objective_mean is not None else main_valid_obj
    )
    hf_like_score = base_objective + cfg.hf.generalization_penalty_weight * generalization_penalty
    if epoch_baseline_violations is not None:
        hf_like_score += cfg.baseline_epoch_violation_weight * float(epoch_baseline_violations)

    logger.info(
        "Free-loss timing: init=%.3fs, train=%.3fs, eval=%.3fs",
        t_init_end - t_init_start,
        t_train_end - t_train_start,
        t_eval_end - t_eval_start,
    )

    return {
        "hf_like_score": hf_like_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "epoch_objective_mean": epoch_objective_mean,
        "epoch_baseline_violations": epoch_baseline_violations,
        "epoch_better_than_baseline": epoch_better_than_baseline,
        "epoch_eval": {
            "enabled": bool(steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch > 0 else None,
            "epochs_total": int(epochs_total),
            "objectives": epoch_validation_objectives,
            "objective_mean": epoch_objective_mean,
            "baseline_margins": epoch_baseline_margins,
            "baseline_violations": epoch_baseline_violations,
            "better_than_baseline": epoch_better_than_baseline,
        },
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "pair_count": int(total_pairs),
        "early_eval": {
            "enabled": bool(early_eval_effective),
            "steps": int(early_eval_effective),
            "baseline_validation_objective": baseline_early_valid,
            "candidate_validation_objective": early_validation_objective,
            "early_stopped": early_stopped,
        },
        "phases": {
            "f1": {
                "steps": int(steps_f1),
                "train_score_mean": float(score_meter_f1.avg) if steps_f1 > 0 else None,
                "train_loss_mean": float(loss_meter_f1.avg) if steps_f1 > 0 else None,
                "pair_count": int(total_pairs_f1),
            },
            "f2": {
                "steps": int(steps_f2),
                "train_score_mean": float(score_meter.avg) if steps_f2 > 0 else None,
                # For f2 we report the overall mean, since it is evaluated on the
                # full trajectory; callers can combine this with the f1 view.
                "train_loss_mean": float(loss_meter.avg) if steps_f2 > 0 else None,
                "pair_count": int(total_pairs - total_pairs_f1),
            },
        },
        "config": {
            "hf": asdict(cfg.hf),
            "free_loss": {
                "f1_steps": cfg.f1_steps,
                "total_train_steps": steps,
                "f2_steps": cfg.f2_steps,
                "f3_enabled": cfg.f3_enabled,
                "baseline_epoch_violation_weight": cfg.baseline_epoch_violation_weight,
            },
        },
        "loss_ir": {
            "name": compiled_loss.ir.name,
            "intuition": compiled_loss.ir.intuition,
            "hyperparams": compiled_loss.ir.hyperparams,
            "operators_used": compiled_loss.ir.operators_used,
        },
    }
