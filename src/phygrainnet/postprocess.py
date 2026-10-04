"""Make predictions structurally valid and combine models in a metric-aware way."""
from __future__ import annotations

import numpy as np


def make_valid(pred: np.ndarray, decimals: int = 4) -> np.ndarray:
    """Clip to [0, 100], enforce non-decreasing, force the 200 mm column to exactly 100.

    Rounding happens before the monotone pass so rounding can never create a
    decrease, and the last column is written as the literal 100.0 required by
    the scorer (softmax+cumsum only gives ~100 up to float error).
    """
    p = np.asarray(pred, dtype=np.float64).copy()
    if p.ndim == 1:
        p = p[None, :]
    p = np.nan_to_num(p, nan=0.0, posinf=100.0, neginf=0.0)
    p = np.clip(p, 0.0, 100.0)
    p = np.round(p, decimals)
    p = np.maximum.accumulate(p, axis=1)
    p[:, -1] = 100.0
    return p


def pointwise_median(preds: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """(Weighted) median per support point.

    The metric is a weighted L1 per point, so the median is the loss-optimal
    point estimate across ensemble members; the pointwise median of
    non-decreasing curves is itself non-decreasing.
    """
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in preds], axis=0)
    if weights is None:
        return np.median(stack, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    order = np.argsort(stack, axis=0)
    sorted_vals = np.take_along_axis(stack, order, axis=0)
    sorted_w = w[order]
    cw = np.cumsum(sorted_w, axis=0)
    idx = np.argmax(cw >= 0.5, axis=0)
    return np.take_along_axis(sorted_vals, idx[None], axis=0)[0]


def shrink_to_prior(pred: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend with a prior curve (e.g. training median); stays monotone."""
    return (1 - alpha) * np.asarray(pred) + alpha * np.asarray(prior)[None, :]


def masses_to_cdf(masses: np.ndarray) -> np.ndarray:
    m = np.clip(np.asarray(masses, dtype=np.float64), 0, None)
    m = m / np.maximum(m.sum(axis=-1, keepdims=True), 1e-12) * 100.0
    return np.cumsum(m, axis=-1)


def cdf_to_masses(cdf: np.ndarray) -> np.ndarray:
    c = np.asarray(cdf, dtype=np.float64)
    return np.diff(np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1), axis=-1)
