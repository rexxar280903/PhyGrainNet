"""Label loading and leakage-safe grouping."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from phygrainnet.constants import SAMPLE_SUBMISSION_FILE, TARGET_COLUMNS, TRAIN_LABELS_FILE


def _canon_col(c: object) -> str:
    """'2.0' -> '2', '200.00' -> '200', ' 0.063 ' -> '0.063'."""
    s = str(c).strip()
    try:
        v = float(s)
    except ValueError:
        return s
    return f"{v:g}"


def load_train_labels(data_root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_root) / TRAIN_LABELS_FILE)
    df.columns = [_canon_col(c) for c in df.columns]
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"label file lacks {missing}; got {list(df.columns)}")
    df["sample_id"] = df["sample_id"].astype(str)
    return df[["sample_id", *TARGET_COLUMNS]].reset_index(drop=True)


def load_test_ids(data_root: str | Path) -> list[str]:
    return pd.read_csv(Path(data_root) / SAMPLE_SUBMISSION_FILE)["sample_id"].astype(str).tolist()


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COLUMNS].to_numpy(dtype=np.float64)


_ID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def site_groups(sample_ids: list[str], max_gap: int = 10) -> np.ndarray:
    """Group ids that probably come from the same site / borehole.

    Rule (DEC-005): same letter prefix and numeric part within `max_gap` of the
    previous id in sorted order -> same group (H366..H374 become one group,
    H181/H183 one group, H615/H616/H617 one group). Ids that do not follow the
    letter+number pattern each get their own group.
    """
    parsed = []
    for i, sid in enumerate(sample_ids):
        m = _ID_RE.match(sid.strip())
        parsed.append((m.group(1).upper(), int(m.group(2)), i) if m else (sid, None, i))
    groups = np.zeros(len(sample_ids), dtype=int)
    gid = -1
    prev = None
    for prefix, num, i in sorted(parsed, key=lambda t: (t[0], -1 if t[1] is None else t[1])):
        if num is None or prev is None or prev[0] != prefix or prev[1] is None or num - prev[1] > max_gap:
            gid += 1
        groups[i] = gid
        prev = (prefix, num)
    return groups


def leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    for g in np.unique(groups):
        val = np.where(groups == g)[0]
        trn = np.where(groups != g)[0]
        folds.append((trn, val))
    return folds


def grouped_kfold(groups: np.ndarray, n_folds: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Greedy balanced grouped K-fold (largest groups first)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    order = sorted(uniq, key=lambda g: -sizes[g])
    bins: list[list[int]] = [[] for _ in range(n_folds)]
    load = np.zeros(n_folds)
    for g in order:
        k = int(np.argmin(load))
        bins[k].append(g)
        load[k] += sizes[g]
    folds = []
    for b in bins:
        val = np.where(np.isin(groups, b))[0]
        trn = np.where(~np.isin(groups, b))[0]
        folds.append((trn, val))
    return folds


def make_folds(sample_ids: list[str], strategy: str = "logo", n_folds: int = 5, seed: int = 42, max_gap: int = 10):
    groups = site_groups(sample_ids, max_gap=max_gap)
    if strategy == "logo":
        return groups, leave_one_group_out(groups)
    if strategy == "group_kfold":
        return groups, grouped_kfold(groups, n_folds, seed)
    raise ValueError(f"unknown fold strategy {strategy!r}")
