# Hankel anomaly detection: production-oriented starter

This is a first refactoring step for the research script. It separates model definitions,
data utilities, training, scoring, and artifact handling, and adds basic tests and CI.

## Important scope note

The original `.mat` data files were not included with the Python script. Therefore, this
starter does **not** claim that the entire experiment has been reproduced end-to-end.
Before connecting the real data, verify the shape and meaning of each MATLAB variable.
In particular, check how the original script uses `N`, `Hank_rank`, `MC`, and the
shapes of `H_M_all`, `H_test_all`, `H_Sd1_all`, and `H_Sd2_all`.

## 1. Set up the environment (Windows PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Use `python -m pytest`, rather than `pytest`, to avoid the PATH problem where the
`pytest` command is not found.

## 2. Run tests

```powershell
python -m pytest -q
```

The tests check basic data utility and anomaly-threshold behavior. They do not validate
the scientific correctness of the full experiment.

## 3. Suggested project layout

- `src/hankel_anomaly/models.py`: neural network architectures
- `src/hankel_anomaly/data.py`: input validation and data loading helpers
- `src/hankel_anomaly/training.py`: reusable training loop
- `src/hankel_anomaly/scoring.py`: anomaly score and threshold functions
- `src/hankel_anomaly/artifacts.py`: model checkpoint save/load
- `tests/`: automated tests
- `.github/workflows/tests.yml`: run checks on every push and pull request

## 4. Scientific checks to complete before deployment

1. Confirm tensor dimensions at each loading and reshaping step.
2. Ensure the threshold is calibrated only from an agreed nominal calibration set.
3. Separate training, calibration, and final test data by experiment/run where possible.
4. Define the intended unit of a score: one row, one Hankel matrix, or one Monte Carlo trial.
5. Evaluate false-positive rate and damage-detection rate, not only a plot.
6. Save the configuration, random seed, dataset version, metrics, and model artifact together.

## 5. Next increments

1. Connect the actual `.mat` files and add a data-shape report.
2. Add a complete training/evaluation entry point.
3. Add MLflow experiment tracking.
4. Add a prediction interface and batch inference tests.
5. Add monitoring and a retraining decision policy.
6. Consider Databricks only after the local workflow is reproducible.
