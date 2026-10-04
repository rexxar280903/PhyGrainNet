"""Small models on per-sample feature vectors (trained from scratch, no pretraining).

Targets are handled as interval masses in centred-log-ratio space so every
prediction maps back to a valid cumulative curve.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phygrainnet.postprocess import cdf_to_masses, make_valid, masses_to_cdf, pointwise_median

EPS_MASS = 0.5  # % added before the log ratio (labels contain exact zeros)


def cdf_to_clr(cdf: np.ndarray) -> np.ndarray:
    m = np.clip(cdf_to_masses(cdf), 0, None) + EPS_MASS
    lg = np.log(m)
    return lg - lg.mean(axis=-1, keepdims=True)


def clr_to_cdf(clr: np.ndarray) -> np.ndarray:
    z = np.exp(clr - clr.max(axis=-1, keepdims=True))
    m = z / z.sum(axis=-1, keepdims=True) * (100.0 + EPS_MASS * clr.shape[-1]) - EPS_MASS
    return masses_to_cdf(np.clip(m, 0, None))


class CLRRegressor:
    """Ridge or PLS in CLR space of the 11 interval masses."""

    def __init__(self, kind: str = "ridge", alpha: float = 10.0, n_components: int = 3):
        self.kind, self.alpha, self.n_components = kind, alpha, n_components

    def fit(self, X: np.ndarray, Y_cdf: np.ndarray) -> "CLRRegressor":
        T = cdf_to_clr(Y_cdf)
        if self.kind == "ridge":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        elif self.kind == "pls":
            nc = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
            self.model = make_pipeline(StandardScaler(), PLSRegression(n_components=nc, scale=False))
        else:
            raise ValueError(self.kind)
        self.model.fit(X, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return make_valid(clr_to_cdf(np.asarray(self.model.predict(X))))


class KNNMedian:
    """Distance-weighted pointwise median of the k nearest training curves."""

    def __init__(self, k: int = 3, power: float = 1.0):
        self.k, self.power = k, power

    def fit(self, X: np.ndarray, Y_cdf: np.ndarray) -> "KNNMedian":
        self.scaler = StandardScaler().fit(X)
        self.X = self.scaler.transform(X)
        self.Y = np.asarray(Y_cdf, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        out = []
        for z in Z:
            d = np.sqrt(((self.X - z) ** 2).sum(axis=1)) / np.sqrt(self.X.shape[1])
            k = min(self.k, len(d))
            idx = np.argsort(d)[:k]
            w = 1.0 / (d[idx] + 1e-3) ** self.power
            out.append(pointwise_median([self.Y[i] for i in idx], list(w)))
        return make_valid(np.array(out))


class PriorMedian:
    """Image-free baseline: pointwise median of training curves."""

    def fit(self, X, Y_cdf):
        self.curve = np.median(np.asarray(Y_cdf, dtype=np.float64), axis=0)
        return self

    def predict(self, X):
        return make_valid(np.repeat(self.curve[None], len(X), axis=0))


class PriorMean(PriorMedian):
    def fit(self, X, Y_cdf):
        self.curve = np.mean(np.asarray(Y_cdf, dtype=np.float64), axis=0)
        return self


def build_classical(spec: dict):
    kind = spec["kind"]
    if kind == "ridge_clr":
        return CLRRegressor("ridge", alpha=float(spec.get("alpha", 10.0)))
    if kind == "pls_clr":
        return CLRRegressor("pls", n_components=int(spec.get("n_components", 3)))
    if kind == "knn_median":
        return KNNMedian(k=int(spec.get("k", 3)), power=float(spec.get("power", 1.0)))
    if kind == "prior_median":
        return PriorMedian()
    if kind == "prior_mean":
        return PriorMean()
    raise ValueError(f"unknown classical model {kind!r}")
