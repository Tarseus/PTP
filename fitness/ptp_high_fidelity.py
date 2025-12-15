from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ptp_dsl import compile_ptp_program, parse_ptp_dsl


def _setup_tsp_imports(repo_root: str) -> None:
    """Configure sys.path so that we can import TSP/POMO modules.

    This mirrors the path manipulation used in the original training scripts,
    but keeps the current working directory unchanged.
    """

    tsp_root = os.path.join(repo_root, "POMO", "TSP")
    tsp_pomo = os.path.join(tsp_root, "POMO")
    utils_root = os.path.join(repo_root, "POMO")

    for path in (tsp_root, tsp_pomo, utils_root):
        if path not in sys.path:
            sys.path.insert(0, path)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_setup_tsp_imports(REPO_ROOT)

from TSPEnv import TSPEnv  # type: ignore  # noqa: E402
from TSPModel import TSPModel  # type: ignore  # noqa: E402
from utils.utils import AverageMeter  # type: ignore  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass
class HighFidelityConfig:
    """Configuration for short-run high-fidelity PTP evaluation."""

    problem: str = "tsp"
    hf_steps: int = 200
    # Optional epoch-style configuration. When both hf_epochs and
    # hf_instances_per_epoch are > 0, they are used to derive the
    # total number of training steps as:
    #   total_steps = hf_epochs * ceil(hf_instances_per_epoch / train_batch_size)
    # and hf_steps is treated as a fallback/legacy setting.
    hf_epochs: int = 0
    hf_instances_per_epoch: int = 0
    train_problem_size: int = 20
    valid_problem_sizes: Sequence[int] = (100,)
    train_batch_size: int = 64
    pomo_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    alpha: float = 0.05  # preference sharpness
    device: str = "cuda"
    seed: int = 0
    num_validation_episodes: int = 128
    validation_batch_size: int = 64
    generalization_penalty_weight: float = 1.0
    pool_version: str = "v0"


def get_total_hf_train_steps(config: HighFidelityConfig) -> int:
    """Return total training *steps* given an HF config.

    - If both hf_epochs and hf_instances_per_epoch are > 0, derive the total
      number of steps from an epoch-style configuration.
    - Otherwise, fall back to hf_steps (legacy behaviour).
    """

    if config.hf_epochs > 0 and config.hf_instances_per_epoch > 0:
        batch_size = max(int(config.train_batch_size), 1)
        steps_per_epoch = math.ceil(config.hf_instances_per_epoch / batch_size)
        total_steps = config.hf_epochs * steps_per_epoch
        return max(int(total_steps), 1)

    return max(int(config.hf_steps), 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_struct_repr_from_tour(tour: torch.Tensor) -> Any:
    """Convert a TSP tour (sequence of node indices) into a structural
    representation suitable for delta_struct computations.
    """

    # tour: (problem_size,)
    nodes = tour.tolist()
    length = len(nodes)
    if length == 0:
        return set()
    edges = set()
    for i in range(length):
        u = int(nodes[i])
        v = int(nodes[(i + 1) % length])
        edges.add((u, v))
    return edges


def _estimate_struct_delta_from_edges(a: Any, b: Any) -> float:
    """Estimate structural delta given two edge-set representations."""

    if not isinstance(a, set) or not isinstance(b, set):
        return 0.0
    if not a and not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return 1.0 - intersection / max(union, 1)


def _train_one_batch_with_ptp(
    env: TSPEnv,
    model: TSPModel,
    optimizer: Adam,
    compiled_ptp,
    config: HighFidelityConfig,
    step_index: int,
) -> Tuple[float, float]:
    """Single training batch for TSP with PTP-defined preferences."""

    batch_size = config.train_batch_size

    # Augmentation is disabled in HF evaluation for simplicity and speed.
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
    # Convert to objective (positive length).
    objective = -reward  # shape: (batch, pomo)

    # Log probabilities per solution.
    log_prob = torch.log(prob_list + 1e-8).sum(dim=2)  # (batch, pomo)

    total_loss = 0.0
    total_pairs = 0

    alpha = float(config.alpha)
    total_steps = config.hf_steps

    # Iterate per-instance to build pools and preferences.
    for batch_index in range(batch_size):
        instance_meta = {
            "size": int(env.problem_size),
            "batch_index": int(batch_index),
        }

        solutions_meta: List[Dict[str, Any]] = []
        struct_repr_list: List[Any] = []
        for pomo_index in range(env.pomo_size):
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

        pool_meta = {"solutions": solutions_meta}
        # Stage support has been removed; use a single global stage identifier.
        stage_name = "main"

        anchor_ids = compiled_ptp.select_anchors(instance_meta, pool_meta, stage_name)
        if not anchor_ids:
            continue

        pairs = compiled_ptp.build_preferences(anchor_ids, pool_meta, stage_name)
        if not pairs:
            continue

        # Map solution_id -> index within this instance.
        index_by_id = {
            int(meta["solution_id"]): idx for idx, meta in enumerate(solutions_meta)
        }

        for winner_id, loser_id in pairs:
            winner_idx = index_by_id.get(int(winner_id))
            loser_idx = index_by_id.get(int(loser_id))
            if winner_idx is None or loser_idx is None:
                continue

            obj_w = solutions_meta[winner_idx]["objective"]
            obj_l = solutions_meta[loser_idx]["objective"]
            delta_obj = float(obj_l - obj_w)

            struct_w = struct_repr_list[winner_idx]
            struct_l = struct_repr_list[loser_idx]
            delta_struct = _estimate_struct_delta_from_edges(struct_w, struct_l)

            pair_size = float(env.problem_size)
            weight = compiled_ptp.weight_preference(
                delta_obj=delta_obj,
                delta_struct=delta_struct,
                size=pair_size,
                stage=stage_name,
            )

            # Skip zero-weight pairs to save computation.
            if weight <= 0.0:
                continue

            logp_w = log_prob[batch_index, winner_idx]
            logp_l = log_prob[batch_index, loser_idx]
            logit_diff = alpha * (logp_w - logp_l)
            pair_loss = -float(weight) * F.logsigmoid(logit_diff)

            total_loss += pair_loss
            total_pairs += 1

    if total_pairs == 0:
        # Fallback to an RL-style loss to keep training stable.
        advantage = reward - reward.mean(dim=1, keepdim=True)
        rl_log_prob = log_prob
        loss = -(advantage * rl_log_prob).mean()
    else:
        loss = total_loss / total_pairs

    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return score_mean.item(), float(loss.item())


@torch.no_grad()
def _evaluate_tsp_model(
    model: TSPModel,
    problem_size: int,
    pomo_size: int,
    device: torch.device,
    num_episodes: int,
    batch_size: int,
) -> float:
    """Evaluate a TSP model by running POMO rollouts on random instances."""

    # TSPModel assumes that when pomo_size > problem_size we have
    # pomo_size % problem_size == 0 for its early-evaluation branch.
    # If that invariant is violated, we fall back to pomo_size <= problem_size
    # to avoid shape mismatches in the model's reshape logic.
    effective_pomo_size = pomo_size
    if pomo_size > problem_size and pomo_size % problem_size != 0:
        logger.warning(
            "Adjusting pomo_size from %d to %d for evaluation (problem_size=%d) "
            "to avoid invalid reshape in TSPModel.",
            pomo_size,
            problem_size,
            problem_size,
        )
        effective_pomo_size = problem_size

    logger.info(
        "Evaluating TSP model: problem_size=%d, pomo_size=%d, episodes=%d, batch_size=%d",
        problem_size,
        effective_pomo_size,
        num_episodes,
        batch_size,
    )

    env = TSPEnv(
        problem_size=problem_size,
        pomo_size=effective_pomo_size,
        device=str(device),
    )
    model.eval()

    score_meter = AverageMeter()
    episodes_done = 0

    while episodes_done < num_episodes:
        remaining = num_episodes - episodes_done
        current_batch = min(batch_size, remaining)

        env.load_problems(current_batch)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, prob = model(state)
            state, reward, done = env.step(selected)

        # reward: (batch, pomo)
        max_pomo_reward, _ = reward.max(dim=1)
        # Convert to objective (positive cost).
        score = (-max_pomo_reward.float()).mean().item()
        score_meter.update(score, n=current_batch)

        logger.debug(
            "Eval batch done: problem_size=%d, pomo_size=%d, batch=%d, "
            "episodes_done=%d/%d, batch_score=%.6f, avg_score=%.6f",
            problem_size,
            effective_pomo_size,
            current_batch,
            episodes_done + current_batch,
            num_episodes,
            score,
            float(score_meter.avg),
        )

        episodes_done += current_batch

    return float(score_meter.avg)


def evaluate_ptp_dsl_high_fidelity(
    ptp_dsl_source: str,
    config: HighFidelityConfig,
) -> Dict[str, Any]:
    """High-fidelity evaluation pipeline for a PTP DSL program.

    Steps:
        1. Compile DSL into executable PTP program.
        2. Run short-run TSP training with PTP-defined preferences.
        3. Evaluate the trained model on validation and larger problem sizes.
        4. Return HF_score and detailed metrics for logging and selection.
    """

    if config.problem != "tsp":
        raise NotImplementedError(
            "Currently only problem='tsp' is supported for PTP HF evaluation."
        )

    _set_seed(config.seed)

    device_str = config.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    parsed_spec = parse_ptp_dsl(ptp_dsl_source)
    compiled_ptp = compile_ptp_program(parsed_spec)

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
        problem_size=config.train_problem_size,
        pomo_size=config.pomo_size,
        device=str(device),
    )

    model = TSPModel(**model_params).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    score_meter = AverageMeter()
    loss_meter = AverageMeter()

    total_steps = get_total_hf_train_steps(config)

    logger.info(
        "Starting HF training: steps=%d, train_problem_size=%d, pomo_size=%d, "
        "batch_size=%d, device=%s",
        total_steps,
        config.train_problem_size,
        config.pomo_size,
        config.train_batch_size,
        str(device),
    )

    log_interval = max(total_steps // 10, 1)

    for step_index in range(total_steps):
        score, loss = _train_one_batch_with_ptp(
            env=env,
            model=model,
            optimizer=optimizer,
            compiled_ptp=compiled_ptp,
            config=config,
            step_index=step_index,
        )
        score_meter.update(score)
        loss_meter.update(loss)

        if (step_index + 1) % log_interval == 0 or step_index == 0:
            logger.info(
                "HF train step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f)",
                step_index + 1,
                total_steps,
                score,
                float(score_meter.avg),
                loss,
                float(loss_meter.avg),
            )

    # Validation on the training scale.
    main_valid_obj = _evaluate_tsp_model(
        model=model,
        problem_size=config.train_problem_size,
        pomo_size=config.pomo_size,
        device=device,
        num_episodes=config.num_validation_episodes,
        batch_size=config.validation_batch_size,
    )

    # Generalization to larger problem sizes.
    gen_objectives: Dict[int, float] = {}
    for size in config.valid_problem_sizes:
        size_int = int(size)
        gen_obj = _evaluate_tsp_model(
            model=model,
            problem_size=size_int,
            pomo_size=config.pomo_size,
            device=device,
            num_episodes=config.num_validation_episodes,
            batch_size=config.validation_batch_size,
        )
        gen_objectives[size_int] = gen_obj

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    hf_score = main_valid_obj + config.generalization_penalty_weight * generalization_penalty

    logger.info(
        "HF evaluation complete: train_size=%d, valid_sizes=%s, "
        "hf_score=%.6f, main_valid_obj=%.6f, gen_penalty=%.6f",
        config.train_problem_size,
        ",".join(str(int(s)) for s in config.valid_problem_sizes),
        hf_score,
        main_valid_obj,
        generalization_penalty,
    )

    return {
        "hf_score": hf_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "config": asdict(config),
        "ptp_dsl": ptp_dsl_source,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-fidelity evaluation for PTP DSL candidates (TSP)."
    )
    parser.add_argument(
        "--dsl_path",
        type=str,
        required=True,
        help="Path to a file containing the PTP DSL JSON program.",
    )
    parser.add_argument(
        "--hf_steps",
        type=int,
        default=200,
        help="Number of short-run training steps for high-fidelity evaluation.",
    )
    parser.add_argument(
        "--hf_epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs for high-fidelity evaluation. "
            "If both --hf_epochs and --hf_instances_per_epoch are > 0, they "
            "are used to derive the total number of training steps instead "
            "of --hf_steps."
        ),
    )
    parser.add_argument(
        "--hf_instances_per_epoch",
        type=int,
        default=0,
        help=(
            "Number of training instances per epoch for high-fidelity "
            "evaluation. Used together with --hf_epochs to derive the total "
            "number of training steps when both are > 0."
        ),
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="tsp",
        help="Problem identifier (currently only 'tsp' is supported).",
    )
    parser.add_argument(
        "--train_problem_size",
        type=int,
        default=20,
        help="Problem size used during short-run training.",
    )
    parser.add_argument(
        "--valid_problem_sizes",
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
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser


def main_cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()

    with open(args.dsl_path, "r", encoding="utf-8") as f:
        ptp_dsl_source = f.read()

    config = HighFidelityConfig(
        problem=args.problem,
        hf_steps=args.hf_steps,
        hf_epochs=args.hf_epochs,
        hf_instances_per_epoch=args.hf_instances_per_epoch,
        train_problem_size=args.train_problem_size,
        valid_problem_sizes=tuple(args.valid_problem_sizes),
        device=args.device,
        seed=args.seed,
    )

    result = evaluate_ptp_dsl_high_fidelity(ptp_dsl_source, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main_cli()
