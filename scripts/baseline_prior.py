"""A000: image-free baselines + the first valid submission (week 1 safety net).

    python scripts/baseline_prior.py --data-root /kaggle/input/soil-grain-size-from-photos \
        --out /kaggle/working/submission_A000_median.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from phygrainnet.data.labels import label_matrix, load_test_ids, load_train_labels, make_folds
from phygrainnet.metrics.kaggle_emd import emd_per_sample
from phygrainnet.submission import write_submission
from phygrainnet.experiment import append_registry, registry_row, write_outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/submission_A000_median.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/outputs"))
    ap.add_argument("--registry", type=Path, default=Path("/kaggle/working/registry.csv"))
    ap.add_argument("--cv", choices=["logo", "group_kfold"], default="group_kfold")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    labels = load_train_labels(args.data_root)
    y = label_matrix(labels)
    ids = labels["sample_id"].tolist()
    test_ids = load_test_ids(args.data_root)
    groups, folds = make_folds(ids, args.cv, args.n_folds, args.seed)
    cfg = {"experiment_id": "A000", "seed": args.seed,
           "data": {"root": str(args.data_root)}, "cv": {"strategy": args.cv, "n_folds": args.n_folds}}
    splits = [{"fold": k, "train_ids": [ids[i] for i in trn], "val_ids": [ids[i] for i in val]}
              for k, (trn, val) in enumerate(folds)]
    for name, fn in (("mean", lambda a: a.mean(0)), ("median", lambda a: np.median(a, 0))):
        oof = np.zeros_like(y)
        for trn, val in folds:
            oof[val] = fn(y[trn])
        pred = np.repeat(fn(y)[None], len(test_ids), 0)
        exp_id = f"A000_{name}"
        summary = write_outputs(args.out_dir, exp_id, ids, y, oof, groups, test_ids, pred, cfg,
                                {"fold_splits": splits})
        append_registry(args.registry, registry_row(exp_id, f"prior_{name}", "baseline_prior.py", args.seed,
                                                    summary, (time.time() - t0) / 60))
        print(f"A000 {name:6s}: grouped {args.cv} CV EMD = {summary['cv_emd']:.3f}")
    equal = np.cumsum(np.full(11, 100 / 11))
    print(f"host equal-mass baseline on train: {emd_per_sample(y, np.repeat(equal[None], len(y), 0)).mean():.3f}")

    pred = np.repeat(np.median(y, 0)[None], len(test_ids), 0)
    write_submission(test_ids, pred, args.out)
    print(f"wrote valid submission -> {args.out}")


if __name__ == "__main__":
    main()
