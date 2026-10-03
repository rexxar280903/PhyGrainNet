from __future__ import annotations

import numpy as np


DEFAULT_DIAMETERS_MM = np.array(
    [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0],
    dtype=np.float64,
)


def provisional_log_grid_emd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    diameters_mm: np.ndarray = DEFAULT_DIAMETERS_MM,
) -> np.ndarray:
    """Provisional log-grid cumulative EMD.

    This must be verified against the live Kaggle evaluation definition before
    it is promoted to the official project metric.
    """
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


def mean_provisional_log_grid_emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(provisional_log_grid_emd(y_true, y_pred)))
