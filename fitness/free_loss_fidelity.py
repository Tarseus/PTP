from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence, Tuple

import logging
import torch
from torch.optim import Adam

from .ptp_high_fidelity import (
    HighFidelityConfig,
    _set_seed,
    _evaluate_tsp_model,
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
        advantage = reward - reward.mean(dim=1, keepdim=True)
        rl_log_prob = log_prob
        loss = -(advantage * rl_log_prob).mean()
    else:
        cost_a_tensor = objective[b_idx, winner_idx]
        cost_b_tensor = objective[b_idx, loser_idx]
        logp_w_tensor = log_prob[b_idx, winner_idx]
        logp_l_tensor = log_prob[b_idx, loser_idx]

        weight = torch.ones_like(cost_a_tensor)
        batch = {
            "cost_a": cost_a_tensor,
            "cost_b": cost_b_tensor,
            "log_prob_w": logp_w_tensor,
            "log_prob_l": logp_l_tensor,
            "weight": weight,
        }
        loss = compiled_loss.loss_fn(batch=batch, model_output={}, extra={"alpha": hf_cfg.alpha})

    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return score_mean.item(), float(loss.item()), pair_count


def evaluate_free_loss_candidate(
    compiled_loss: CompiledFreeLoss,
    cfg: FreeLossFidelityConfig,
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

    score_meter = AverageMeter()
    loss_meter = AverageMeter()
    total_pairs = 0

    steps = max(int(cfg.f1_steps), 1)
    logger.info(
        "Free-loss training: steps=%d, train_problem_size=%d, pomo_size=%d, batch_size=%d, device=%s",
        steps,
        cfg.hf.train_problem_size,
        cfg.hf.pomo_size,
        cfg.hf.train_batch_size,
        str(device),
    )

    log_interval = max(steps // 10, 1)

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

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    hf_like_score = main_valid_obj + cfg.hf.generalization_penalty_weight * generalization_penalty

    return {
        "hf_like_score": hf_like_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "pair_count": int(total_pairs),
        "config": {
            "hf": asdict(cfg.hf),
            "free_loss": {
                "f1_steps": cfg.f1_steps,
                "f2_steps": cfg.f2_steps,
                "f3_enabled": cfg.f3_enabled,
            },
        },
        "loss_ir": {
            "name": compiled_loss.ir.name,
            "intuition": compiled_loss.ir.intuition,
            "hyperparams": compiled_loss.ir.hyperparams,
            "operators_used": compiled_loss.ir.operators_used,
        },
    }
