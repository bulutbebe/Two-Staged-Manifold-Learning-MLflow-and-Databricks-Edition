"""Train the two-stage Hankel anomaly model and track the run in MLflow."""

import argparse
from pathlib import Path
import sys

# Make the src-layout package importable when this script is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import mlflow.pytorch
import torch

from hankel_anomaly.artifacts import save_checkpoint
from hankel_anomaly.config import TrainingConfig
from hankel_anomaly.data import load_mat_tensor, prepare_ambient_training_data
from hankel_anomaly.pipeline import train_two_stage
from hankel_anomaly.tracking import configure_mlflow

DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
EXPERIMENT_NAME = "/Shared/hankel-anomaly-detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the two-stage Hankel anomaly model and log the run to MLflow."
    )
    parser.add_argument(
        "--margin-struct",
        type=float,
        default=1000.0,
        help="Ambient-space bound expansion factor (default: 1000). Use 0 for no expansion.",
    )
    args = parser.parse_args()
    if args.margin_struct < 0:
        parser.error("--margin-struct must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    tracking_uri = configure_mlflow(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")

    nominal = load_mat_tensor(
        DATA_DIR / "H_strongNL_MC.mat", "H_nom_strong_all"
    )
    print(f"Nominal training data: {tuple(nominal.shape)} ({nominal.dtype})")

    config = TrainingConfig(
        latent_dim=50,
        hidden_dim=1024,
        epochs_round1=10,
        epochs_round2=10,
        learning_rate=1e-4,
        batch_size=78,  # matches the original research script
        seed=24,
    )
    n_monte_carlo = 1000
    margin_struct = args.margin_struct
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"two-stage-margin-{margin_struct:g}-seed-{config.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "epochs_round1": config.epochs_round1,
            "epochs_round2": config.epochs_round2,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "threshold_std_multiplier": config.threshold_std_multiplier,
            "n_monte_carlo": n_monte_carlo,
            "margin_struct": margin_struct,
            "input_dim": int(nominal.shape[1]),
            "nominal_n_rows": int(nominal.shape[0]),
            "device": device_name,
        })
        mlflow.set_tag("project", "hankel-anomaly-detection")
        mlflow.set_tag("training_method", "two-stage-complementary-autoencoder")

        ambient = prepare_ambient_training_data(
            nominal,
            n_monte_carlo=n_monte_carlo,
            margin_struct=margin_struct,
            seed=config.seed,
        )
        print(f"Ambient training data: {tuple(ambient.shape)} ({ambient.dtype})")
        print(f"Training on: {device_name.upper()}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("Starting two-stage training...")

        result = train_two_stage(nominal, ambient, config=config)
        metadata = {
            "input_dim": result["input_dim"],
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "epochs_round1": config.epochs_round1,
            "epochs_round2": config.epochs_round2,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "threshold_std_multiplier": config.threshold_std_multiplier,
            "n_monte_carlo": n_monte_carlo,
            "margin_struct": margin_struct,
            "nominal_shape": list(nominal.shape),
            "ambient_shape": list(ambient.shape),
            "mlflow_run_id": run.info.run_id,
        }
        round1_path = ARTIFACT_DIR / "round1_model.pt"
        round2_path = ARTIFACT_DIR / "round2_model.pt"
        save_checkpoint(round1_path, result["model_round1"], metadata)
        save_checkpoint(round2_path, result["model_round2"], metadata)

        for epoch, loss in enumerate(result["history_round1"], start=1):
            mlflow.log_metric("round1_train_loss", float(loss), step=epoch)
        for epoch, loss in enumerate(result["history_round2"], start=1):
            mlflow.log_metric("round2_train_loss", float(loss), step=epoch)
        mlflow.log_metrics({
            "round1_final_loss": float(result["history_round1"][-1]),
            "round2_final_loss": float(result["history_round2"][-1]),
        })

        # Keep local checkpoints for the existing workflow and also store copies in MLflow.
        # These checkpoint files include the state_dict (learned weights and biases)
        # plus metadata, and are convenient for the existing evaluation script.
        mlflow.log_artifact(str(round1_path), artifact_path="models")
        mlflow.log_artifact(str(round2_path), artifact_path="models")
        mlflow.log_dict(metadata, "model_metadata.json")

        # Log each trained stage as a first-class MLflow PyTorch model.
        # This stores the model weights together with MLflow model metadata, making
        # each stage independently identifiable and loadable from the MLflow run.
        mlflow.pytorch.log_model(
            pytorch_model=result["model_round1"].cpu().eval(),
            name="round1_model",
            serialization_format="pickle",
        )

        mlflow.pytorch.log_model(
            pytorch_model=result["model_round2"].cpu().eval(),
            name="round2_model",
            serialization_format="pickle",
        )

        print("Training complete.")
        print(f"Round 1 final loss: {result['history_round1'][-1]:.6g}")
        print(f"Round 2 final loss: {result['history_round2'][-1]:.6g}")
        print(f"Saved model checkpoints to: {ARTIFACT_DIR}")
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
