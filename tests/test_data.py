
import numpy as np
import pytest
import torch

from hankel_anomaly.data import (
    flatten_hankel_merged,
    load_mat_tensor,
    sample_hypercube,
)


def test_flatten_hankel_merged():
    matrix = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    result = flatten_hankel_merged(matrix)

    assert result.shape == (2, 9)
    assert torch.equal(result[0], matrix[:3].reshape(-1))
    assert torch.equal(result[1], matrix[3:].reshape(-1))

def test_flatten_rejects_invalid_shape():
    matrix = torch.ones((5, 3))

    with pytest.raises(ValueError):
        flatten_hankel_merged(matrix)

def test_sample_hypercube_respects_bounds():
    lower = torch.tensor([0.0, 10.0])
    upper = torch.tensor([1.0, 20.0])
    generator = torch.Generator().manual_seed(42)

    samples = sample_hypercube(
        lower, upper, n_samples=100, generator=generator
    )

    assert samples.shape == (100, 2)
    assert torch.all(samples >= lower)
    assert torch.all(samples <= upper)

def test_sample_hypercube_rejects_invalid_bounds():
    lower = torch.tensor([2.0, 0.0])
    upper = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError):
        sample_hypercube(lower, upper, n_samples=10)

def test_load_mat_tensor(tmp_path):
    path = tmp_path / "example.mat"
    original = np.arange(6, dtype=np.float64).reshape(2, 3)

    from scipy.io import savemat
    savemat(path, {"example_data": original})

    result = load_mat_tensor(path, "example_data")

    assert result.shape == (2, 3)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor(original, dtype=torch.float32))

def test_load_mat_tensor_rejects_missing_variable(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "example.mat"
    savemat(path, {"other_data": np.ones((2, 2))})

    with pytest.raises(KeyError):
        load_mat_tensor(path, "missing_data")

def test_prepare_ambient_training_data_preserves_shape_and_is_reproducible():
    from hankel_anomaly.data import prepare_ambient_training_data

    # Four vertically stacked 2x2 blocks; 4 Monte Carlo draws make the
    # flattened sample count match the merged matrix's element count.
    nominal = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    margin_struct = 1

    ambient1 = prepare_ambient_training_data(
        nominal,
        n_monte_carlo=4,
        margin_struct=margin_struct,
        seed=7,
    )
    ambient2 = prepare_ambient_training_data(
        nominal,
        n_monte_carlo=4,
        margin_struct=margin_struct,
        seed=7,
    )

    assert ambient1.shape == nominal.shape
    assert torch.equal(ambient1, ambient2)

    flattened = flatten_hankel_merged(nominal)
    value_min = flattened.amin(dim=0)
    value_max = flattened.amax(dim=0)
    value_range = value_max - value_min

    lower = value_min - margin_struct * value_range
    upper = value_max + margin_struct * value_range

    # Each sampled coordinate must lie within its expanded bounds.
    sampled_flat = flatten_hankel_merged(ambient1)
    assert torch.all(sampled_flat >= lower)
    assert torch.all(sampled_flat <= upper)

def test_prepare_ambient_training_data_zero_margin_uses_original_bounds():
    from hankel_anomaly.data import prepare_ambient_training_data

    nominal = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    ambient = prepare_ambient_training_data(
        nominal, n_monte_carlo=4, margin_struct=0, seed=7
    )

    flattened = flatten_hankel_merged(nominal)
    lower = flattened.amin(dim=0)
    upper = flattened.amax(dim=0)
    samples = ambient.reshape(4, 4)
    assert torch.all(samples >= lower)
    assert torch.all(samples <= upper)


def test_prepare_ambient_training_data_rejects_negative_margin():
    from hankel_anomaly.data import prepare_ambient_training_data

    nominal = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    with pytest.raises(ValueError, match="margin_struct"):
        prepare_ambient_training_data(nominal, n_monte_carlo=4, margin_struct=-1)