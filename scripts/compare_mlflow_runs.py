"""
Compare Hankel anomaly-detection training and evaluation runs recorded in MLflow.

Run from the project root after configuring Databricks MLflow environment variables:
    python scripts/compare_mlflow_runs.py

Optional:
    python scripts/compare_mlflow_runs.py --experiment "/Shared/hankel-anomaly-detection"
    python scripts/compare_mlflow_runs.py --output-dir artifacts/comparisons

The script reads recorded MLflow runs; it does not retrain or change models.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


DEFAULT_EXPERIMENT = "/Shared/hankel-anomaly-detection"

TRAIN_PREFIX = "two-stage-margin-"
EVAL_PREFIX = "evaluate-margin-"

TRAIN_METRICS = [
    "round1_final_loss",
    "round2_final_loss",
    "round1_train_loss",
    "round2_train_loss",
]

EVAL_METRICS = [
    "calibration_score_mean",
    "calibration_score_std",
    "anomaly_threshold",
    "nominal_false_positive_rate",
    "damage1_detection_rate",
    "damage2_detection_rate",
]


def get_seed(run_name: str, params: dict) -> str:
    """Read seed from a run name such as two-stage-margin-100-seed-24."""
    match = re.search(r"-seed-([^-]+)$", run_name or "")
    if match:
        return match.group(1)
    return str(params.get("seed", "unknown"))


def get_margin(run) -> float | None:
    """Read margin_struct from MLflow parameters, with run-name fallback."""
    value = run.data.params.get("margin_struct")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"(?:two-stage|evaluate)-margin-([-+]?\d*\.?\d+)", run.info.run_name or "")
    if match:
        return float(match.group(1))
    return None


def runs_to_frame(runs, prefix: str, metrics: list[str]) -> pd.DataFrame:
    """Convert MLflow Run objects to a small table and keep latest successful run per key."""
    records = []

    for run in runs:
        name = run.info.run_name or ""
        if not name.startswith(prefix):
            continue

        margin = get_margin(run)
        if margin is None:
            continue

        record = {
            "margin_struct": margin,
            "seed": get_seed(name, run.data.params),
            "run_name": name,
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
        }

        for metric in metrics:
            record[metric] = run.data.metrics.get(metric, float("nan"))

        records.append(record)

    columns = ["margin_struct", "seed", "run_name", "run_id", "status", "start_time", *metrics]
    frame = pd.DataFrame(records, columns=columns)

    if frame.empty:
        return frame

    # If the same margin/seed has been run more than once, use the most recent FINISHED run.
    frame = frame[frame["status"] == "FINISHED"].copy()
    frame = frame.sort_values("start_time")
    frame = frame.drop_duplicates(["margin_struct", "seed"], keep="last")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", default="artifacts/comparisons")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Uses the current MLflow environment, e.g. MLFLOW_TRACKING_URI=databricks.
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {args.experiment}\n"
            "Check MLFLOW_TRACKING_URI and the experiment path."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )

    training = runs_to_frame(runs, TRAIN_PREFIX, TRAIN_METRICS)
    evaluation = runs_to_frame(runs, EVAL_PREFIX, EVAL_METRICS)

    if training.empty:
        raise RuntimeError("No FINISHED training runs found with the expected run-name prefix.")
    if evaluation.empty:
        raise RuntimeError("No FINISHED evaluation runs found with the expected run-name prefix.")

    # Join on both configuration and seed, so runs from different seeds are not mixed.
    comparison = training.merge(
        evaluation,
        on=["margin_struct", "seed"],
        how="outer",
        suffixes=("_training", "_evaluation"),
        indicator=True,
    ).sort_values(["margin_struct", "seed"], ascending=[False, True])

    csv_path = output_dir / "mlflow_model_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    display_columns = [
        "margin_struct",
        "seed",
        "_merge",
        "round1_final_loss",
        "round2_final_loss",
        "nominal_false_positive_rate",
        "damage1_detection_rate",
        "damage2_detection_rate",
        "anomaly_threshold",
    ]
    display_columns = [col for col in display_columns if col in comparison.columns]

    print(f"\nExperiment: {args.experiment}")
    print(f"Finished training configurations: {len(training)}")
    print(f"Finished evaluation configurations: {len(evaluation)}")
    print("\nComparison (training and evaluation matched by margin_struct + seed):")
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")

    # Plot training losses.
    loss_columns = [
        ("round1_final_loss", "Round 1 final loss"),
        ("round2_final_loss", "Round 2 final loss"),
    ]
    fig, ax = plt.subplots()
    plotted = False
    for column, label in loss_columns:
        if column in training and training[column].notna().any():
            grouped = training.groupby("margin_struct")[column].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", label=label)
            plotted = True
    if plotted:
        ax.set_xlabel("margin_struct")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss by margin_struct")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = output_dir / "training_losses.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {path}")
    plt.close(fig)

    # Plot evaluation rates as percentages.
    rate_columns = [
        ("nominal_false_positive_rate", "Nominal false-positive rate"),
        ("damage1_detection_rate", "Damage 1 detection rate"),
        ("damage2_detection_rate", "Damage 2 detection rate"),
    ]
    fig, ax = plt.subplots()
    plotted = False
    for column, label in rate_columns:
        if column in evaluation and evaluation[column].notna().any():
            grouped = evaluation.groupby("margin_struct")[column].mean().sort_index()
            ax.plot(grouped.index, grouped.values * 100, marker="o", label=label)
            plotted = True
    if plotted:
        ax.set_xlabel("margin_struct")
        ax.set_ylabel("Rate (%)")
        ax.set_title("Evaluation rates by margin_struct")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = output_dir / "evaluation_rates.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {path}")
    plt.close(fig)

    unmatched = comparison[comparison["_merge"] != "both"]
    if not unmatched.empty:
        print(
            "\nNote: Some training/evaluation runs did not match by both margin_struct and seed. "
            "Check the CSV rows marked 'left_only' or 'right_only'."
        )


if __name__ == "__main__":
    main()
