"""Geotechnical diagnostics and site-cluster bootstrap on complete OOF runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phygrainnet.constants import DIAMETERS_MM, TARGET_COLUMNS
from phygrainnet.metrics.kaggle_emd import emd_per_sample


def valid_curves(curves) -> np.ndarray:
    x = np.asarray(curves, dtype=float)
    if x.ndim != 2 or x.shape[1] != 11 or not len(x):
        raise ValueError("expected nonempty (N, 11) curves")
    if (not np.isfinite(x).all() or (x < 0).any() or (x > 100).any()
            or (np.diff(x, axis=1) < -1e-6).any() or not np.allclose(x[:, -1], 100, atol=1e-4)):
        raise ValueError("invalid cumulative curves")
    return x


def fractions(curves) -> np.ndarray:
    """Clay <0.002, silt 0.002–0.063, sand 0.063–2, gravel 2–200 mm."""
    x = valid_curves(curves)
    return np.stack([x[:, 0], x[:, 3] - x[:, 0], x[:, 6] - x[:, 3], 100 - x[:, 6]], axis=1)


def diameter_at(curve, percent: float) -> float | None:
    """Invert piecewise log-linear CDF; leftmost crossing on exact plateaus.

    No extrapolation below the smallest measured sieve. None means censored.
    """
    x = valid_curves(np.asarray(curve)[None])[0]
    if not 0 < percent < 100:
        raise ValueError("percent must be between 0 and 100")
    if percent < x[0]:
        return None
    j = int(np.searchsorted(x, percent, side="left"))
    if x[j] == percent:
        return float(DIAMETERS_MM[j])
    a = (percent - x[j - 1]) / (x[j] - x[j - 1])
    return float(10 ** ((1 - a) * np.log10(DIAMETERS_MM[j - 1]) + a * np.log10(DIAMETERS_MM[j])))


def cluster_interval(values, groups, repeats: int = 2000, seed: int = 42) -> dict:
    """Resample groups with replacement; retain sample-weighted Kaggle mean."""
    v, g = np.asarray(values, dtype=float), np.asarray(groups)
    if v.ndim != 1 or len(v) != len(g) or not len(v) or not np.isfinite(v).all() or repeats < 1:
        raise ValueError("finite matching values/groups and positive repeats required")
    unique = np.unique(g)
    rng = np.random.default_rng(seed)
    bags = [v[g == key] for key in unique]
    means = [float(np.concatenate([bags[i] for i in rng.integers(0, len(bags), len(bags))]).mean()) for _ in range(repeats)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"estimate": float(v.mean()), "lower": float(lo), "upper": float(hi), "level": 0.95,
            "clusters": len(unique), "repeats": repeats, "seed": seed,
            "note": "site-cluster bootstrap of fixed OOF predictions; excludes model refitting and selection uncertainty"}


def load_run(folder: str | Path) -> dict:
    path = Path(folder)
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("cv_complete") is not True:
        raise ValueError(f"{path}: incomplete CV; smoke runs cannot become research evidence")
    frames = {name: pd.read_csv(path / f"{name}.csv", dtype={"sample_id": str}) for name in ["oof", "targets", "groups"]}
    ids = frames["targets"]["sample_id"].tolist()
    for name, frame in frames.items():
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(ids):
            raise ValueError(f"{path}: duplicate or mismatched {name} IDs")
        frames[name] = frame.set_index("sample_id").loc[ids]
    if metrics.get("n_train", len(ids)) != len(ids):
        raise ValueError(f"{path}: missing OOF rows")
    if frames["groups"]["group"].isna().any():
        raise ValueError(f"{path}: missing groups")
    folds = metrics.get("fold_splits", [])
    if not folds:
        raise ValueError(f"{path}: missing fold provenance")
    assignments, seen = [], []
    mapping = frames["groups"]["group"].to_dict()
    for fold in folds:
        trn, val = fold["train_ids"], fold["val_ids"]
        if set(trn) & set(val) or set(trn) | set(val) != set(ids):
            raise ValueError(f"{path}: invalid fold coverage")
        if {mapping[i] for i in trn} & {mapping[i] for i in val}:
            raise ValueError(f"{path}: group leakage")
        seen.extend(val)
        assignments.append({"train_ids": sorted(trn), "val_ids": sorted(val)})
    if sorted(seen) != sorted(ids):
        raise ValueError(f"{path}: OOF coverage must be exactly once")
    return {"id": metrics.get("experiment_id", path.name), "ids": ids, "metrics": metrics,
            "y": valid_curves(frames["targets"][TARGET_COLUMNS]), "p": valid_curves(frames["oof"][TARGET_COLUMNS]),
            "groups": frames["groups"]["group"].to_numpy(), "folds": assignments}


def summarize_run(run: dict, repeats: int = 2000) -> dict:
    y, p, g = run["y"], run["p"], run["groups"]
    per = emd_per_sample(y, p)
    quantiles = {}
    for q in [10, 30, 50, 60]:
        true = [diameter_at(c, q) for c in y]
        pred = [diameter_at(c, q) for c in p]
        pairs = [(a, b) for a, b in zip(true, pred) if a is not None and b is not None]
        quantiles[f"D{q}"] = {"n_defined_pairs": len(pairs), "n_censored_pairs": len(y) - len(pairs),
            "mae_mm": float(np.mean([abs(a - b) for a, b in pairs])) if pairs else None,
            "mae_log10_mm": float(np.mean([abs(np.log10(a) - np.log10(b)) for a, b in pairs])) if pairs else None}
    return {"experiment_id": run["id"], "n_oof": len(y), "cv_emd": float(per.mean()),
            "ci95": cluster_interval(per, g, repeats),
            "fraction_mae_pp": dict(zip(["clay", "silt", "sand", "gravel"], np.abs(fractions(y) - fractions(p)).mean(axis=0).tolist())),
            "quantiles": quantiles, "per_sample": dict(zip(run["ids"], per.tolist())),
            "per_group": {str(k): float(per[g == k].mean()) for k in np.unique(g)},
            "curves": [{"sample_id": sid, "target": a.tolist(), "prediction": b.tolist()} for sid, a, b in zip(run["ids"], y, p)],
            "provenance": {"git_commit": run["metrics"].get("git_commit"),
                           "source_manifest_sha256": run["metrics"].get("source_manifest_sha256"),
                           "source_manifest_matches": run["metrics"].get("source_manifest_matches"),
                           "working_tree_dirty": run["metrics"].get("working_tree_dirty"), "folds": run["folds"]}}


def compare_runs(candidate: dict, reference: dict, repeats: int = 2000) -> dict:
    if candidate["ids"] != reference["ids"]:
        raise ValueError("comparison requires matching sample order; export runs with the same labels")
    if not np.array_equal(candidate["groups"], reference["groups"]) or not np.allclose(candidate["y"], reference["y"]):
        raise ValueError("comparison requires identical groups and targets")
    norm = lambda r: sorted((tuple(f["train_ids"]), tuple(f["val_ids"])) for f in r["folds"])
    if norm(candidate) != norm(reference):
        raise ValueError("comparison requires identical fold splits")
    delta = emd_per_sample(candidate["y"], candidate["p"]) - emd_per_sample(reference["y"], reference["p"])
    g = candidate["groups"]
    return {"candidate": candidate["id"], "reference": reference["id"], "delta_emd_ci95": cluster_interval(delta, g, repeats),
            "groups_improved": sum(float(delta[g == k].mean()) < 0 for k in np.unique(g)), "n_groups": len(np.unique(g)),
            "note": "negative delta favors candidate; paired fixed OOF bootstrap is exploratory after selection"}
