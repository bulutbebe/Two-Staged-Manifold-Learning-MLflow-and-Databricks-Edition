"""Connect validated data tensors to the two-stage training workflow."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig
from .models import AEHankelRound1, AEHankelRound2
from .training import train_autoencoder


def make_dataloader(
    data: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 24,
) -> DataLoader:
    """Build a DataLoader from a 2-D tensor shaped (samples, features)."""
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor")
    if data.ndim != 2:
        raise ValueError("data must be a 2-D tensor with shape (n_samples, n_features)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("data must contain at least one sample and one feature")
    if not torch.is_floating_point(data):
        raise ValueError("data must use a floating-point dtype")
    if not torch.isfinite(data).all():
        raise ValueError("data contains NaN or infinite values")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    generator = torch.Generator().manual_seed(seed)
    # TensorDataset ensures each batch is returned as a one-item tuple, which
    # train_autoencoder already supports.
    return DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_two_stage(
    nominal_data: torch.Tensor,
    ambient_data: torch.Tensor,
    config: TrainingConfig | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train Round 1 on nominal data and Round 2 on ambient-space data.

    Both inputs must be 2-D tensors shaped (samples, features), with the same
    feature count. This function deliberately does not generate ambient data:
    its construction must match the experiment's intended mathematics.
    """
    config = config or TrainingConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, data in (("nominal_data", nominal_data), ("ambient_data", ambient_data)):
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if data.ndim != 2:
            raise ValueError(f"{name} must have shape (n_samples, n_features)")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"{name} must not be empty")
        if not torch.is_floating_point(data):
            raise ValueError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(data).all():
            raise ValueError(f"{name} contains NaN or infinite values")

    if nominal_data.shape[1] != ambient_data.shape[1]:
        raise ValueError(
            "nominal_data and ambient_data must have the same number of features; "
            f"got {nominal_data.shape[1]} and {ambient_data.shape[1]}"
        )
    input_dim = nominal_data.shape[1]
    if not 0 < config.latent_dim < input_dim:
        raise ValueError(
            f"latent_dim must be between 1 and input_dim - 1; got "
            f"latent_dim={config.latent_dim}, input_dim={input_dim}"
        )

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    nominal_loader = make_dataloader(
        nominal_data, config.batch_size, seed=config.seed
    )
    ambient_loader = make_dataloader(
        ambient_data, config.batch_size, seed=config.seed
    )

    model_round1 = AEHankelRound1(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )
    history_round1 = train_autoencoder(
        model_round1,
        nominal_loader,
        epochs=config.epochs_round1,
        learning_rate=config.learning_rate,
        device=device,
        label="round 1",
    )

    # Round 2 shares/fixes the relevant Round 1 parameters, as defined in models.py.
    model_round2 = AEHankelRound2(
        ae1=model_round1,
        latent_dim=config.latent_dim,
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
    )
    history_round2 = train_autoencoder(
        model_round2,
        ambient_loader,
        epochs=config.epochs_round2,
        learning_rate=config.learning_rate,
        device=device,
        label="round 2",
    )

    return {
        "model_round1": model_round1,
        "model_round2": model_round2,
        "history_round1": history_round1,
        "history_round2": history_round2,
        "input_dim": input_dim,
        "device": device,
    }
