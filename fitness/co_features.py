from __future__ import annotations

from typing import Dict, Tuple

import torch


def zscore(x: torch.Tensor, *, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


def rank01(x: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Return normalized rank in [0, 1], where 0 is best (lowest objective)."""

    if x.numel() == 0:
        return x

    k = int(x.shape[dim])
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.float32)

    # argsort(argsort(x)) yields ranks where 0 corresponds to the smallest value.
    order = x.argsort(dim=dim)
    ranks = order.argsort(dim=dim).to(dtype=torch.float32)
    return ranks / float(k - 1)


def robust_scale_mad(x: torch.Tensor, *, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Robust scale estimate using MAD (median absolute deviation)."""

    median = x.median(dim=dim, keepdim=True).values
    mad = (x - median).abs().median(dim=dim, keepdim=True).values
    # For normal distributions, MAD * 1.4826 ~= std.
    scale = mad * 1.4826
    return scale.clamp_min(eps)


def regret(
    objective: torch.Tensor,
    *,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dimensionless regret relative to the best-of-K in the set."""

    best = objective.min(dim=dim, keepdim=True).values
    scale = robust_scale_mad(objective, dim=dim, eps=eps)
    return (objective - best) / scale


def compute_co_features(
    objective: torch.Tensor,
    *,
    dim: int = 1,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute CO-aligned, dimensionless features over a set of K solutions."""

    return {
        "obj_z": zscore(objective, dim=dim, eps=eps),
        "rank": rank01(objective, dim=dim),
        "regret": regret(objective, dim=dim, eps=eps),
    }


def gather_pairwise_deltas(
    features: Dict[str, torch.Tensor],
    *,
    b_idx: torch.Tensor,
    winner_idx: torch.Tensor,
    loser_idx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Gather (loser - winner) deltas for any supported feature."""

    out: Dict[str, torch.Tensor] = {}
    for base_key in ("obj_z", "rank", "regret"):
        if base_key not in features:
            continue
        feat = features[base_key]
        w = feat[b_idx, winner_idx]
        l = feat[b_idx, loser_idx]
        out[f"delta_{'z' if base_key == 'obj_z' else base_key}"] = l - w
    return out


def build_model_output(
    *,
    objective: torch.Tensor,
    log_prob: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (model_output, co_features) for reuse by setwise/pairwise losses."""

    feats = compute_co_features(objective, dim=1, eps=eps)
    model_output: Dict[str, torch.Tensor] = {
        "objective": objective,
        "log_prob": log_prob,
        **feats,
    }
    return model_output, feats

