"""Train the two-stage Hankel anomaly model on local MATLAB data."""

from pathlib import Path
import sys

# Make the src-layout package importable when this script is run from the project root
# without requiring an editable install first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from hankel_anomaly.artifacts import save_checkpoint
from hankel_anomaly.config import TrainingConfig
from hankel_anomaly.data import load_mat_tensor, prepare_ambient_training_data
from hankel_anomaly.pipeline import train_two_stage

DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"


def main() -> None:
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

    ambient = prepare_ambient_training_data(
        nominal, n_monte_carlo=1000, margin_struct=5, seed=config.seed
    )
    print(f"Ambient training data: {tuple(ambient.shape)} ({ambient.dtype})")
    print(f"Training on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
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
        "nominal_shape": list(nominal.shape),
        "ambient_shape": list(ambient.shape),
    }
    save_checkpoint(ARTIFACT_DIR / "round1_model.pt", result["model_round1"], metadata)
    save_checkpoint(ARTIFACT_DIR / "round2_model.pt", result["model_round2"], metadata)

    print("Training complete.")
    print(f"Round 1 final loss: {result['history_round1'][-1]:.6g}")
    print(f"Round 2 final loss: {result['history_round2'][-1]:.6g}")
    print(f"Saved model checkpoints to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
