"""Evaluate saved models on nominal and damaged MATLAB test matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hankel_anomaly.artifacts import load_checkpoint
from hankel_anomaly.data import load_mat_tensor
from hankel_anomaly.evaluation import calibrate_threshold, score_matrix_trials
from hankel_anomaly.models import AEHankelRound1, AEHankelRound2
from hankel_anomaly.scoring import classify_scores

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"

DATASETS = {
    "nominal": ("H_test_strongNL_MC.mat", "H_nom_strong_test_all"),
    "damage1": ("H_s_dmg1_MC.mat", "H_anom_dmg1_strong"),
    "damage2": ("H_s_dmg2_MC.mat", "H_anom_dmg2_strong"),
}


def _checkpoint_metadata(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("metadata", {})


def load_trained_models(device: torch.device):
    """Rebuild the exact model architecture used by the training script and load weights."""
    round1_path = MODEL_DIR / "round1_model.pt"
    round2_path = MODEL_DIR / "round2_model.pt"
    for path in (round1_path, round2_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. Run scripts/train_model.py first."
            )

    metadata = _checkpoint_metadata(round1_path)
    input_dim = int(metadata.get("input_dim", 78))
    latent_dim = int(metadata.get("latent_dim", 50))
    hidden_dim = int(metadata.get("hidden_dim", 1024))

    round1 = AEHankelRound1(input_dim, latent_dim, hidden_dim)
    load_checkpoint(round1_path, round1, map_location="cpu")
    round2 = AEHankelRound2(round1, latent_dim, input_dim, hidden_dim)
    load_checkpoint(round2_path, round2, map_location="cpu")
    round1.to(device).eval()
    round2.to(device).eval()
    return round2, metadata


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    nominal_train = load_mat_tensor(DATA_DIR / "H_strongNL_MC.mat", "H_nom_strong_all")
    print(f"Nominal training data: {tuple(nominal_train.shape)}")
    model, metadata = load_trained_models(device)

    std_multiplier = float(metadata.get("threshold_std_multiplier", 2.0))
    calibration_scores, threshold = calibrate_threshold(
        model,
        nominal_train,
        std_multiplier=std_multiplier,
        batch_size=32,
        device=device,
    )
    nominal_mean = calibration_scores.mean()
    nominal_std = calibration_scores.std(unbiased=False)
    print(f"Calibration blocks: {calibration_scores.numel()}")
    print(f"Calibration score mean: {nominal_mean.item():.8g}")
    print(f"Calibration score std:  {nominal_std.item():.8g}")
    print(f"Anomaly threshold:      {threshold.item():.8g}")

    score_sets: dict[str, torch.Tensor] = {}
    for label, (filename, variable) in DATASETS.items():
        data = load_mat_tensor(DATA_DIR / filename, variable)
        if data.ndim != 3:
            raise ValueError(f"{filename}:{variable} should be 3-D, got {tuple(data.shape)}")
        if data.shape[0] != data.shape[1] or data.shape[0] != int(metadata.get("input_dim", 78)):
            raise ValueError(
                f"Unexpected matrix shape {tuple(data.shape)} in {filename}; "
                "expected (rank, rank, trials) with rank matching model input_dim."
            )
        scores = score_matrix_trials(model, data, batch_size=32, device=device)
        score_sets[label] = scores
        np.save(RESULT_DIR / f"scores_{label}.npy", scores.numpy())
        print(f"{label}: {scores.numel()} trials; score range "
              f"[{scores.min().item():.8g}, {scores.max().item():.8g}]")

    np.save(RESULT_DIR / "nominal_calibration_scores.npy", calibration_scores.numpy())
    np.save(RESULT_DIR / "threshold.npy", np.asarray(threshold.item(), dtype=np.float64))

    summary = {
        "threshold": float(threshold.item()),
        "threshold_rule": (
            f"mean(nominal training block scores) + {std_multiplier:g} * population_std"
        ),
        "calibration_score_mean": float(nominal_mean.item()),
        "calibration_score_population_std": float(nominal_std.item()),
        "device": str(device),
        "model_metadata": metadata,
        "datasets": {},
    }
    rows = []
    for label, scores in score_sets.items():
        predictions = classify_scores(scores, threshold)
        n = int(scores.numel())
        flagged = int(predictions.sum().item())
        stats = {
            "n_trials": n,
            "mean_score": float(scores.mean().item()),
            "std_score_population": float(scores.std(unbiased=False).item()),
            "min_score": float(scores.min().item()),
            "max_score": float(scores.max().item()),
            "flagged_as_anomaly": flagged,
            "flagged_fraction": flagged / n,
        }
        summary["datasets"][label] = stats
        rows.append({"dataset": label, **stats})
        print(f"{label}: {flagged}/{n} flagged ({100 * flagged / n:.2f}%)")

    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (RESULT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [("nominal", "Nominal test"), ("damage1", "Damage 1"), ("damage2", "Damage 2")]
    fig, ax = plt.subplots(figsize=(14, 6))
    cursor = 0
    for key, display in labels:
        values = score_sets[key].numpy()
        x = np.arange(cursor, cursor + len(values))
        ax.scatter(x, values, s=10, alpha=0.7, label=display)
        cursor += len(values)
    ax.axhline(threshold.item(), color="red", linestyle="--", linewidth=2, label="Threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Test trial (grouped by dataset)")
    ax.set_ylabel("Squared complementary representation norm")
    ax.set_title("Hankel anomaly scores")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "anomaly_scores.png", dpi=160)
    plt.close(fig)

    print(f"\nEvaluation results saved to: {RESULT_DIR}")
    print("Files: scores_*.npy, nominal_calibration_scores.npy, threshold.npy, "
          "summary.json, summary.csv, anomaly_scores.png")


if __name__ == "__main__":
    main()
