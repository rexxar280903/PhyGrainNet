"""Official competition metric (verified against the Kaggle Evaluation page, 2026-10-04).

EMD = sum_{i=1..10} |F_i - F^_i| * (log10(x_{i+1}) - log10(x_i))

The final score is the mean over samples, range [0, 500], lower is better.
Note the left-point rule: the 200 mm value never contributes.
"""
from __future__ import annotations

import numpy as np

from phygrainnet.constants import DIAMETERS_MM

DEFAULT_DIAMETERS_MM = DIAMETERS_MM


def emd_per_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    diameters_mm: np.ndarray = DEFAULT_DIAMETERS_MM,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diameters_mm = np.asarray(diameters_mm, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.shape[-1] != diameters_mm.size:
        raise ValueError("last dimension must match diameter support")
    if np.any(diameters_mm <= 0) or np.any(np.diff(diameters_mm) <= 0):
        raise ValueError("diameters must be positive and strictly increasing")

    log_width = np.diff(np.log10(diameters_mm))
    cumulative_error = np.abs(y_true[..., :-1] - y_pred[..., :-1])
    return np.sum(cumulative_error * log_width, axis=-1)


def competition_emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean official EMD over samples (the leaderboard number)."""
    return float(np.mean(emd_per_sample(y_true, y_pred)))


def per_point_contribution(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean contribution of each of the 10 left points to the EMD (diagnostics)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.diff(np.log10(DEFAULT_DIAMETERS_MM))
    return np.mean(np.abs(y_true[:, :-1] - y_pred[:, :-1]) * w, axis=0)


# Backwards-compatible names (the provisional implementation turned out to be exact).
provisional_log_grid_emd = emd_per_sample
mean_provisional_log_grid_emd = competition_emd
