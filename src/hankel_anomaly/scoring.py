"""Anomaly scoring and threshold estimation."""
import torch


def complementary_score(model, samples: torch.Tensor) -> torch.Tensor:
    """Return one squared L2 complementary-latent score per input sample."""
    model.eval()
    with torch.inference_mode():
        _, _, z_complementary = model(samples)
        return torch.sum(z_complementary ** 2, dim=-1)


def estimate_threshold(nominal_scores: torch.Tensor, std_multiplier: float = 2.0) -> torch.Tensor:
    """Mean-plus-k-standard-deviations threshold from nominal calibration scores."""
    scores = nominal_scores.reshape(-1)
    if scores.numel() < 2:
        raise ValueError("At least two nominal calibration scores are required")
    if not torch.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values")
    return scores.mean() + std_multiplier * scores.std(unbiased=False)


def classify_scores(scores: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Return True where the anomaly score exceeds the threshold."""
    return scores > threshold
