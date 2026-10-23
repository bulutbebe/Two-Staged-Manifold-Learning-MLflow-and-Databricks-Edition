import pytest
import torch

from hankel_anomaly.config import TrainingConfig
from hankel_anomaly.pipeline import make_dataloader, train_two_stage


def test_make_dataloader_batches_samples_and_features():
    data = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    loader = make_dataloader(data, batch_size=2)
    batches = list(loader)

    assert [batch[0].shape for batch in batches] == [
        (2, 4), (2, 4), (1, 4)
    ]


def test_make_dataloader_rejects_non_2d_data():
    with pytest.raises(ValueError, match="2-D"):
        make_dataloader(torch.ones(4), batch_size=2)


def test_two_stage_training_runs_on_small_synthetic_data():
    torch.manual_seed(7)
    nominal = torch.randn(12, 6)
    ambient = torch.randn(10, 6)
    config = TrainingConfig(
        latent_dim=2,
        hidden_dim=8,
        epochs_round1=1,
        epochs_round2=1,
        learning_rate=1e-3,
        batch_size=4,
        seed=7,
    )

    result = train_two_stage(
        nominal,
        ambient,
        config=config,
        device=torch.device("cpu"),
    )

    assert len(result["history_round1"]) == 1
    assert len(result["history_round2"]) == 1
    assert result["input_dim"] == 6


def test_two_stage_training_rejects_mismatched_feature_counts():
    with pytest.raises(ValueError, match="same number of features"):
        train_two_stage(
            torch.randn(5, 6),
            torch.randn(5, 7),
            config=TrainingConfig(latent_dim=2, hidden_dim=8),
            device=torch.device("cpu"),
        )
