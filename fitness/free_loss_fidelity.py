from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence, Tuple

import logging
import torch
from torch.optim import Adam

from .ptp_high_fidelity import (
    HighFidelityConfig,
    _build_struct_repr_from_tour,
    _estimate_struct_delta_from_edges,
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
    env: TSPEnv,
    objective: torch.Tensor,
) -> Tuple[Sequence[Mapping[str, Any]], Sequence[Tuple[int, int]], Sequence[Any]]:
    batch_size = objective.size(0)
    pomo_size = objective.size(1)

    all_pairs: list[Tuple[int, int]] = []
    batch_meta: list[Dict[str, Any]] = []
    struct_repr_list: list[Any] = []

    for batch_index in range(batch_size):
        instance_meta = {
            "size": int(env.problem_size),
            "batch_index": int(batch_index),
        }
        batch_meta.append(instance_meta)

        solutions_meta: list[Dict[str, Any]] = []
        for pomo_index in range(pomo_size):
            obj_value = float(objective[batch_index, pomo_index].item())
            tour = env.selected_node_list[batch_index, pomo_index]
            struct_repr = _build_struct_repr_from_tour(tour)
            solutions_meta.append(
                {
                    "solution_id": int(pomo_index),
                    "objective": obj_value,
                    "size": int(env.problem_size),
                    "struct_repr": struct_repr,
                }
            )
            struct_repr_list.append(struct_repr)

        sorted_indices = sorted(
            range(pomo_size),
            key=lambda idx: float(objective[batch_index, idx].item()),
        )
        for better_rank, better_idx in enumerate(sorted_indices):
            for worse_idx in sorted_indices[better_rank + 1 :]:
                all_pairs.append((better_idx, worse_idx))

    return batch_meta, all_pairs, struct_repr_list


def _train_one_batch_with_free_loss(
    env: TSPEnv,
    model: TSPModel,
    optimizer: Adam,
    compiled_loss: CompiledFreeLoss,
    hf_cfg: HighFidelityConfig,
) -> Tuple[float, float]:
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

    objective = -reward
    log_prob = torch.log(prob_list + 1e-8).sum(dim=2)

    _, pairs, _ = _build_preference_pairs(env, objective)

    if not pairs:
        advantage = reward - reward.mean(dim=1, keepdim=True)
        rl_log_prob = log_prob
        loss = -(advantage * rl_log_prob).mean()
    else:
        costs_a = []
        costs_b = []
        logp_w = []
        logp_l = []

        for (winner_idx, loser_idx) in pairs:
            costs_a.append(objective[:, winner_idx])
            costs_b.append(objective[:, loser_idx])
            logp_w.append(log_prob[:, winner_idx])
            logp_l.append(log_prob[:, loser_idx])

        cost_a_tensor = torch.stack(costs_a, dim=0).mean(dim=1)
        cost_b_tensor = torch.stack(costs_b, dim=0).mean(dim=1)
        logp_w_tensor = torch.stack(logp_w, dim=0).mean(dim=1)
        logp_l_tensor = torch.stack(logp_l, dim=0).mean(dim=1)

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

    return score_mean.item(), float(loss.item())


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
        score, loss = _train_one_batch_with_free_loss(
            env=env,
            model=model,
            optimizer=optimizer,
            compiled_loss=compiled_loss,
            hf_cfg=cfg.hf,
        )
        score_meter.update(score)
        loss_meter.update(loss)

        if (step + 1) % log_interval == 0 or step == 0:
            logger.info(
                "Free-loss step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f)",
                step + 1,
                steps,
                score,
                float(score_meter.avg),
                loss,
                float(loss_meter.avg),
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
