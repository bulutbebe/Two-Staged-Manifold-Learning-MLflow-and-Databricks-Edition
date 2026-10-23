import pytest
import torch

from hankel_anomaly.data import flatten_hankel_merged, sample_hypercube
from hankel_anomaly.scoring import classify_scores, estimate_threshold


def test_threshold_and_classification():
    nominal = torch.tensor([1.0, 1.0, 1.0, 1.0])
    threshold = estimate_threshold(nominal)
    result = classify_scores(torch.tensor([0.5, 1.0, 1.1]), threshold)
    assert result.tolist() == [False, False, True]


def test_threshold_requires_two_values():
    with pytest.raises(ValueError):
        estimate_threshold(torch.tensor([1.0]))


def test_hypercube_shape_and_bounds():
    lower = torch.tensor([-1.0, 0.0])
    upper = torch.tensor([1.0, 2.0])
    samples = sample_hypercube(lower, upper, 100, torch.Generator().manual_seed(1))
    assert samples.shape == (100, 2)
    assert torch.all(samples >= lower)
    assert torch.all(samples <= upper)


def test_flatten_hankel_blocks():
    matrix = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    flattened = flatten_hankel_merged(matrix)
    assert flattened.shape == (2, 4)
    assert torch.equal(flattened[0], torch.tensor([0., 1., 2., 3.]))


def test_invalid_hankel_shape():
    with pytest.raises(ValueError):
        flatten_hankel_merged(torch.ones(3, 2))
