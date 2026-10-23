import torch
from torch import nn

from hankel_anomaly.evaluation import calibrate_threshold, score_matrix_trials


class SimpleComplementaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z_comp = x[..., :1] * self.weight
        return x, x[..., :1], z_comp


def test_score_matrix_trials_sums_complementary_norm_per_matrix():
    model = SimpleComplementaryModel()
    matrices = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.0], [0.0, 0.0]],
    ])
    scores = score_matrix_trials(model, matrices)
    assert torch.allclose(scores, torch.tensor([5.0, 4.0]))


def test_calibrate_threshold_scores_merged_blocks():
    model = SimpleComplementaryModel()
    # Two stacked 2x2 blocks; only the first column is used by this test model.
    merged = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    scores, threshold = calibrate_threshold(model, merged, std_multiplier=2.0)
    assert torch.allclose(scores, torch.tensor([5.0, 25.0]))
    expected = scores.mean() + 2 * scores.std(unbiased=False)
    assert torch.allclose(threshold, expected)
