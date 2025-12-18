from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import yaml

# Make this file usable both as a module (`-m ptp_discovery.sanity_check_free_loss_eval`)
# and as a direct script (`python ptp_discovery/sanity_check_free_loss_eval.py`).
if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from fitness.free_loss_fidelity import (  # noqa: E402
    FreeLossFidelityConfig,
    evaluate_free_loss_candidate,
)
from fitness.ptp_high_fidelity import HighFidelityConfig  # noqa: E402
from ptp_discovery.free_loss_compiler import (  # noqa: E402
    CompiledFreeLoss,
    compile_free_loss,
)
from ptp_discovery.free_loss_ir import (  # noqa: E402
    FreeLossImplementationHint,
    FreeLossIR,
)
from ptp_discovery.free_loss_eoh_loop import _get_available_devices  # noqa: E402


LOGGER = logging.getLogger("ptp_discovery.sanity_free_loss_eval")


def _build_hf_configs_from_yaml(
    config_path: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[HighFidelityConfig, FreeLossFidelityConfig, List[str]]:
    """Mirror the HF/free-loss config construction from run_free_loss_eoh."""

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml: Dict[str, Any] = yaml.safe_load(f)

    overrides = overrides or {}
    cfg_yaml.update({k: v for k, v in overrides.items() if v is not None})

    seed = int(cfg_yaml.get("seed", 0))

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

    operator_whitelist = list(cfg_yaml.get("operator_whitelist", []))

    return hf_cfg, free_cfg, operator_whitelist


def _build_sanity_ir(operator_whitelist: List[str]) -> FreeLossIR:
    """Construct a simple, known-good free-loss IR.

    This intentionally matches the fallback template in compile_free_loss:
    it consumes (cost_a, cost_b, log_prob_w, log_prob_l, weight) and
    applies a rank-gap/logsigmoid-style loss. The goal is to exercise the
    same training code path as discovery, without requiring an LLM.
    """

    impl = FreeLossImplementationHint(
        expects=[
            "cost_a",
            "cost_b",
            "log_prob_w",
            "log_prob_l",
            "weight",
        ],
        returns="scalar",
    )

    return FreeLossIR(
        name="sanity_rank_gap_logsigmoid",
        intuition=(
            "Sanity-check free loss: logistic preference over log-probability "
            "differences, with a simple rank-gap modulation."
        ),
        pseudocode=(
            "For each pair, compute cost_gap = cost_b - cost_a; normalize cost_gap "
            "across the batch; form x = alpha * (log_prob_w - log_prob_l) - margin * cost_gap_z; "
            "use -logsigmoid(x) as the per-pair loss and average."
        ),
        hyperparams={},
        operators_used=operator_whitelist or [
            "logsigmoid",
            "zscore",
            "rank_gap",
        ],
        implementation_hint=impl,
        code="",
        theoretical_basis="sanity_check_only",
    )


def _run_sequential_sanity_check(
    *,
    hf_cfg: HighFidelityConfig,
    free_cfg: FreeLossFidelityConfig,
    compiled_loss: CompiledFreeLoss,
    num_candidates: int,
    device_override: str | None = None,
    empty_cache: bool = False,
) -> None:
    """Synchronously evaluate several candidates and monitor GPU memory."""

    base_device_str = device_override or hf_cfg.device
    devices = _get_available_devices(base_device_str)
    if not devices:
        devices = [base_device_str]

    LOGGER.info("Sanity check devices: %s", ", ".join(devices))

    for i in range(num_candidates):
        device_str = devices[i % len(devices)]

        # Clone HF config with overridden device.
        hf_local_dict: Dict[str, Any] = dict(hf_cfg.__dict__)
        hf_local_dict["device"] = device_str
        hf_local = HighFidelityConfig(**hf_local_dict)
        free_local = FreeLossFidelityConfig(
            hf=hf_local,
            f1_steps=free_cfg.f1_steps,
            f2_steps=free_cfg.f2_steps,
            f3_enabled=free_cfg.f3_enabled,
        )

        LOGGER.info(
            "Running candidate %d/%d on device=%s (train_problem_size=%d, batch_size=%d, pomo_size=%d)",
            i + 1,
            num_candidates,
            device_str,
            hf_local.train_problem_size,
            hf_local.train_batch_size,
            hf_local.pomo_size,
        )

        try:
            _ = evaluate_free_loss_candidate(
                compiled_loss,
                free_local,
                baseline_early_valid=None,
                early_eval_steps=0,
            )
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - runtime safety
            LOGGER.error(
                "CUDA OOM while evaluating candidate %d on device %s: %s",
                i + 1,
                device_str,
                exc,
            )
            raise

        if torch.cuda.is_available():
            dev = torch.device(device_str if device_str != "cuda" else "cuda:0")
            torch.cuda.synchronize(dev)
            allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
            LOGGER.info(
                "After candidate %d on %s: allocated=%.3f GiB, reserved=%.3f GiB",
                i + 1,
                dev,
                allocated,
                reserved,
            )
            if empty_cache:
                torch.cuda.empty_cache()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check script: synchronously evaluate several free-loss "
            "candidates to probe GPU memory usage."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (same format as free_loss_discovery.yaml).",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="Number of synthetic candidates to evaluate sequentially.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--empty-cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each candidate.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()

    hf_cfg, free_cfg, operator_whitelist = _build_hf_configs_from_yaml(
        args.config,
        overrides={"device": args.device} if args.device is not None else None,
    )
    ir = _build_sanity_ir(operator_whitelist)
    compiled = compile_free_loss(ir, operator_whitelist=operator_whitelist)

    _run_sequential_sanity_check(
        hf_cfg=hf_cfg,
        free_cfg=free_cfg,
        compiled_loss=compiled,
        num_candidates=int(args.num_candidates),
        device_override=args.device,
        empty_cache=bool(args.empty_cache),
    )


if __name__ == "__main__":
    main()

