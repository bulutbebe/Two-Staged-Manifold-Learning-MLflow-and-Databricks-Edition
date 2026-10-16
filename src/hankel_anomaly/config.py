"""Central place for experiment settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    latent_dim: int = 50
    hidden_dim: int = 1024
    epochs_round1: int = 10
    epochs_round2: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 1
    seed: int = 24
    threshold_std_multiplier: float = 2.0
