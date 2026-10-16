"""Data-loading helpers. Adapt MAT variable names to the supplied datasets."""
from pathlib import Path
import numpy as np
import scipy.io
import torch


def load_mat_tensor(path: str | Path, variable_name: str) -> torch.Tensor:
    """Load a named MATLAB variable and return a float32 tensor."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path.resolve()}")
    contents = scipy.io.loadmat(path)
    if variable_name not in contents:
        available = [key for key in contents if not key.startswith("__")]
        raise KeyError(f"Variable {variable_name!r} not found in {path.name}. "
                       f"Available variables: {available}")
    array = np.asarray(contents[variable_name])
    if not np.isfinite(array).all():
        raise ValueError(f"{variable_name!r} contains NaN or infinite values")
    return torch.as_tensor(array, dtype=torch.float32)


def sample_hypercube(lower: torch.Tensor, upper: torch.Tensor, n_samples: int,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample uniformly within a box defined by lower and upper bounds."""
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape")
    if torch.any(upper < lower):
        raise ValueError("upper bounds must be >= lower bounds")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    random_values = torch.rand(
        (n_samples, lower.numel()), dtype=lower.dtype, device=lower.device,
        generator=generator
    )
    return lower.unsqueeze(0) + random_values * (upper - lower).unsqueeze(0)


def flatten_hankel_merged(hankel: torch.Tensor) -> torch.Tensor:
    """Flatten each rank-by-rank block from a vertically merged Hankel matrix.

    Expected input shape: (number_of_blocks * rank, rank).
    """
    if hankel.ndim != 2 or hankel.shape[0] % hankel.shape[1] != 0:
        raise ValueError("Expected a 2-D tensor with rows divisible by column count")
    rank = hankel.shape[1]
    n_blocks = hankel.shape[0] // rank
    return hankel.reshape(n_blocks, rank * rank)
