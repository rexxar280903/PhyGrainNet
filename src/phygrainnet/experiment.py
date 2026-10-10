"""Experiment bookkeeping: output folders, OOF/test files, registry rows."""
from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import importlib.metadata
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from phygrainnet.constants import TARGET_COLUMNS
from phygrainnet.metrics.kaggle_emd import emd_per_sample, per_point_contribution

REGISTRY_COLUMNS = [
    "experiment_id", "date", "architecture", "config", "seed", "fold", "cv_emd", "cv_emd_std",
    "mae", "parameters", "runtime_min", "git_commit", "status", "notes",
]


def git_commit(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def curves_frame(ids: list[str], curves: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(curves), columns=TARGET_COLUMNS)
    df.insert(0, "sample_id", ids)
    return df


def summarize_oof(ids: list[str], y: np.ndarray, oof: np.ndarray, groups: np.ndarray) -> dict:
    per = emd_per_sample(y, oof)
    by_group = pd.Series(per).groupby(groups).mean()
    return {
        "cv_emd": float(per.mean()),
        "cv_emd_median": float(np.median(per)),
        "cv_emd_group_mean": float(by_group.mean()),
        "cv_emd_std_over_groups": float(by_group.std(ddof=0)),
        "mae": float(np.mean(np.abs(y[:, :-1] - oof[:, :-1]))),
        "worst_samples": {ids[i]: float(per[i]) for i in np.argsort(-per)[:5]},
        "per_point": {c: float(v) for c, v in zip(TARGET_COLUMNS[:-1], per_point_contribution(y, oof))},
        "per_sample": {i: float(v) for i, v in zip(ids, per)},
        "per_group": {str(k): float(v) for k, v in by_group.items()},
    }


def write_outputs(
    out_dir: str | Path,
    exp_id: str,
    train_ids: list[str],
    y: np.ndarray,
    oof: np.ndarray,
    groups: np.ndarray,
    test_ids: list[str],
    test_pred: np.ndarray,
    cfg: dict,
    extra: dict | None = None,
) -> dict:
    out = Path(out_dir) / exp_id
    out.mkdir(parents=True, exist_ok=True)
    curves_frame(train_ids, oof).to_csv(out / "oof.csv", index=False)
    curves_frame(test_ids, test_pred).to_csv(out / "test.csv", index=False)
    curves_frame(train_ids, y).to_csv(out / "targets.csv", index=False)
    pd.DataFrame({"sample_id": train_ids, "group": groups}).to_csv(out / "groups.csv", index=False)
    summary = summarize_oof(train_ids, y, oof, groups)
    summary.update(extra or {})
    summary["experiment_id"] = exp_id
    summary["git_commit"] = git_commit()
    summary.setdefault("cv_complete", True)
    summary["n_oof"] = len(train_ids)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    packages = {}
    for name in ("numpy", "pandas", "torch", "scikit-learn", "scikit-image", "opencv-python-headless", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    (out / "environment.json").write_text(json.dumps({"python": platform.python_version(),
        "platform": platform.platform(), "packages": packages}, indent=2), encoding="utf-8")
    return summary


def append_registry(path: str | Path, row: dict) -> None:
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_COLUMNS})


def registry_row(exp_id: str, architecture: str, config_path: str, seed: int, summary: dict, runtime_min: float,
                 parameters: int | str = "", notes: str = "") -> dict:
    return {
        "experiment_id": exp_id,
        "date": dt.date.today().isoformat(),
        "architecture": architecture,
        "config": config_path,
        "seed": seed,
        "fold": "oof",
        "cv_emd": round(summary["cv_emd"], 4),
        "cv_emd_std": round(summary["cv_emd_std_over_groups"], 4),
        "mae": round(summary["mae"], 4),
        "parameters": parameters,
        "runtime_min": round(runtime_min, 2),
        "git_commit": summary.get("git_commit", git_commit()),
        "status": "done" if summary.get("cv_complete", True) else "smoke_only",
        "notes": notes,
    }
