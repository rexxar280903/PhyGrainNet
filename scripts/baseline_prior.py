"""A000: image-free baselines + the first valid submission (week 1 safety net).

    python scripts/baseline_prior.py --data-root /kaggle/input/soil-grain-size-from-photos \
        --out /kaggle/working/submission_A000_median.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from phygrainnet.data.labels import label_matrix, load_test_ids, load_train_labels, make_folds
from phygrainnet.metrics.kaggle_emd import emd_per_sample
from phygrainnet.submission import write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/submission_A000_median.csv"))
    args = ap.parse_args()

    labels = load_train_labels(args.data_root)
    y = label_matrix(labels)
    ids = labels["sample_id"].tolist()
    groups, folds = make_folds(ids, "logo")
    for name, fn in (("mean", lambda a: a.mean(0)), ("median", lambda a: np.median(a, 0))):
        oof = np.zeros_like(y)
        for trn, val in folds:
            oof[val] = fn(y[trn])
        print(f"A000 {name:6s}: grouped LOGO CV EMD = {emd_per_sample(y, oof).mean():.3f}")
    equal = np.cumsum(np.full(11, 100 / 11))
    print(f"host equal-mass baseline on train: {emd_per_sample(y, np.repeat(equal[None], len(y), 0)).mean():.3f}")

    test_ids = load_test_ids(args.data_root)
    pred = np.repeat(np.median(y, 0)[None], len(test_ids), 0)
    write_submission(test_ids, pred, args.out)
    print(f"wrote valid submission -> {args.out}")


if __name__ == "__main__":
    main()
