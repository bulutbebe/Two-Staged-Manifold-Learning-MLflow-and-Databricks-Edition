"""Evaluation helpers for the original block-wise Hankel anomaly score."""

from __future__ import annotations

import torch

from .scoring import estimate_threshold


def score_matrix_trials(
    model,
    matrices: torch.Tensor,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Score matrices stored as (rank, rank, n_trials).

    For each trial, the score is the squared Frobenius/L2 norm of the
    complementary representation over all rows and complementary coordinates,
    matching the original research script's ``R.T @ R`` calculation.
    """
    if not isinstance(matrices, torch.Tensor) or matrices.ndim != 3:
        raise ValueError("matrices must be a 3-D tensor shaped (rank, rank, n_trials)")
    if matrices.shape[0] != matrices.shape[1]:
        raise ValueError("each test matrix must be square")
    if matrices.shape[2] == 0:
        raise ValueError("matrices must contain at least one trial")
    if not torch.is_floating_point(matrices) or not torch.isfinite(matrices).all():
        raise ValueError("matrices must contain finite floating-point values")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    device = device or next(model.parameters()).device
    trials = matrices.permute(2, 0, 1).contiguous()
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, trials.shape[0], batch_size):
            batch = trials[start : start + batch_size].to(device)
            # The research model concatenates latent components along dim=1, so
            # pass rows as a 2-D batch and restore the matrix grouping afterward.
            n_trials, rank, n_features = batch.shape
            _, _, z_complementary = model(batch.reshape(n_trials * rank, n_features))
            z_complementary = z_complementary.reshape(n_trials, rank, -1)
            # One score per matrix, summing over its rows and complementary dimensions.
            scores = z_complementary.square().sum(dim=(1, 2))
            result.append(scores.detach().cpu())
    return torch.cat(result)


def score_nominal_calibration_blocks(
    model,
    nominal_merged: torch.Tensor,
    *,
    block_size: int | None = None,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Score vertically merged nominal data one square Hankel block at a time."""
    if not isinstance(nominal_merged, torch.Tensor) or nominal_merged.ndim != 2:
        raise ValueError("nominal_merged must be a 2-D tensor")
    rank = nominal_merged.shape[1]
    block_size = rank if block_size is None else block_size
    if block_size != rank:
        raise ValueError("block_size must equal the number of columns (Hankel rank)")
    if nominal_merged.shape[0] == 0 or nominal_merged.shape[0] % rank:
        raise ValueError("nominal_merged rows must be a positive multiple of its rank")
    if not torch.is_floating_point(nominal_merged) or not torch.isfinite(nominal_merged).all():
        raise ValueError("nominal_merged must contain finite floating-point values")

    blocks = nominal_merged.reshape(-1, rank, rank).permute(1, 2, 0).contiguous()
    return score_matrix_trials(model, blocks, batch_size=batch_size, device=device)


def calibrate_threshold(
    model,
    nominal_merged: torch.Tensor,
    *,
    std_multiplier: float = 2.0,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-block nominal calibration scores, mean + k*population-std)."""
    scores = score_nominal_calibration_blocks(
        model, nominal_merged, batch_size=batch_size, device=device
    )
    return scores, estimate_threshold(scores, std_multiplier=std_multiplier)
