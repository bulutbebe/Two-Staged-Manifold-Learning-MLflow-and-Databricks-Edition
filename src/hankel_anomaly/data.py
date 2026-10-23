"""Data-loading and preprocessing helpers for Hankel anomaly detection."""

from pathlib import Path

import numpy as np
import scipy.io
import torch


def load_mat_tensor(
    path: str | Path,
    variable_name: str,
) -> torch.Tensor:
    """Load a numeric MATLAB variable and return a float32 tensor."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path.resolve()}")

    contents = scipy.io.loadmat(path)
    if variable_name not in contents:
        available = [key for key in contents if not key.startswith("__")]
        raise KeyError(
            f"Variable {variable_name!r} not found in {path.name}. "
            f"Available variables: {available}"
        )

    array = np.asarray(contents[variable_name])
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"{variable_name!r} must contain numeric data; got dtype {array.dtype}"
        )
    if np.iscomplexobj(array):
        raise ValueError(
            f"{variable_name!r} contains complex values; "
            "this pipeline currently expects real-valued data"
        )
    if array.size == 0:
        raise ValueError(f"{variable_name!r} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{variable_name!r} contains NaN or infinite values")

    return torch.as_tensor(array, dtype=torch.float32)


def sample_hypercube(
    lower: torch.Tensor,
    upper: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniformly between coordinate-wise lower and upper bounds."""
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower and upper must be one-dimensional tensors")
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape")
    if lower.numel() == 0:
        raise ValueError("Bounds must contain at least one coordinate")
    if lower.device != upper.device or lower.dtype != upper.dtype:
        raise ValueError("lower and upper must have the same device and dtype")
    if not torch.is_floating_point(lower):
        raise ValueError("Bounds must use a floating-point dtype")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("Bounds must contain only finite values")
    if torch.any(upper < lower):
        raise ValueError("upper bounds must be greater than or equal to lower bounds")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    random_values = torch.rand(
        (n_samples, lower.numel()),
        dtype=lower.dtype,
        device=lower.device,
        generator=generator,
    )
    return lower.unsqueeze(0) + random_values * (upper - lower).unsqueeze(0)

def flatten_hankel_merged(hankel: torch.Tensor) -> torch.Tensor:
    """Flatten each rank-by-rank block in a vertically merged matrix.

    Expected input shape: (number_of_blocks * rank, rank).
    Returns shape: (number_of_blocks, rank * rank).
    """
    if hankel.ndim != 2:
        raise ValueError("Expected a two-dimensional tensor")

    rows, columns = hankel.shape
    if columns == 0:
        raise ValueError("Hankel matrix must have at least one column")
    if rows == 0:
        raise ValueError("Hankel matrix must have at least one row")
    if rows % columns != 0:
        raise ValueError("Expected rows to be divisible by the number of columns")
    if not torch.is_floating_point(hankel):
        raise ValueError("Hankel matrix must be floating-point data")
    if not torch.isfinite(hankel).all():
        raise ValueError("Hankel matrix contains NaN or infinite values")

    rank = columns
    n_blocks = rows // rank
    return hankel.reshape(n_blocks, rank * rank)

def prepare_ambient_training_data(
    nominal_data: torch.Tensor,
    n_monte_carlo: int = 1000,
    margin_struct: float = 1000.0,
    seed: int = 24,
) -> torch.Tensor:
    """Reproduce the original ambient-space sampling from merged nominal data.

    ``nominal_data`` must have shape (n_blocks * rank, rank). The function
    flattens each rank-by-rank block to estimate coordinate-wise bounds, draws
    expands the coordinate-wise bounds by ``margin_struct`` times each
    coordinate's original range, draws ``n_monte_carlo`` points in the
    expanded hypercube, then reshapes them back to the same two-dimensional
    shape as ``nominal_data``. Set ``margin_struct=0`` to use the unexpanded
    bounds.
    """
    if not isinstance(nominal_data, torch.Tensor):
        raise TypeError("nominal_data must be a torch.Tensor")
    if nominal_data.ndim != 2:
        raise ValueError("nominal_data must be a 2-D tensor")
    if nominal_data.shape[0] == 0 or nominal_data.shape[1] == 0:
        raise ValueError("nominal_data must not be empty")
    if not torch.is_floating_point(nominal_data):
        raise ValueError("nominal_data must use a floating-point dtype")
    if not torch.isfinite(nominal_data).all():
        raise ValueError("nominal_data contains NaN or infinite values")
    if nominal_data.shape[0] % nominal_data.shape[1] != 0:
        raise ValueError(
            "nominal_data rows must be divisible by its number of columns "
            "to represent vertically merged square blocks"
        )
    if not isinstance(n_monte_carlo, int) or isinstance(n_monte_carlo, bool) or n_monte_carlo < 1:
        raise ValueError("n_monte_carlo must be a positive integer")
    if (
        not isinstance(margin_struct, (int, float))
        or isinstance(margin_struct, bool)
        or not np.isfinite(margin_struct)
        or margin_struct < 0
    ):
        raise ValueError("margin_struct must be a finite, non-negative number")

    flattened = flatten_hankel_merged(nominal_data)
    lower = flattened.amin(dim=0)
    upper = flattened.amax(dim=0)
    value_range = upper - lower
    lower = lower - margin_struct * value_range
    upper = upper + margin_struct * value_range
    generator = torch.Generator(device=nominal_data.device).manual_seed(seed)
    ambient_flat = sample_hypercube(
        lower, upper, n_samples=n_monte_carlo, generator=generator
    )

    expected_elements = nominal_data.numel()
    if ambient_flat.numel() != expected_elements:
        raise ValueError(
            "Ambient sample count is incompatible with nominal_data shape: "
            f"{ambient_flat.numel()} sampled values versus {expected_elements} "
            "values required. Choose n_monte_carlo = nominal_data.shape[0] "
            "/ nominal_data.shape[1]."
        )
    return ambient_flat.reshape_as(nominal_data)
