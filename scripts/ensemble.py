"""H-series: combine experiments by pointwise median, chosen greedily on OOF EMD.

    python scripts/ensemble.py --runs outputs/A001_ridge outputs/C100 outputs/D200 \
        --data-root /kaggle/input/soil-grain-size-from-photos --out submission_H001.csv

Every run directory needs oof.csv and test.csv (written by scripts/train.py).
Only samples present in every run's OOF are used for selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from phygrainnet.constants import TARGET_COLUMNS
from phygrainnet.data.labels import label_matrix, load_test_ids, load_train_labels
from phygrainnet.metrics.kaggle_emd import emd_per_sample
from phygrainnet.postprocess import make_valid, pointwise_median
from phygrainnet.submission import read_prediction_csv, write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-members", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="use every run instead of greedy selection")
    args = ap.parse_args()

    labels = load_train_labels(args.data_root).set_index("sample_id")
    test_ids = load_test_ids(args.data_root)
    oofs, tests = {}, {}
    for r in args.runs:
        oofs[r.name] = read_prediction_csv(r / "oof.csv").set_index("sample_id")
        tests[r.name] = read_prediction_csv(r / "test.csv").set_index("sample_id").loc[test_ids]
    common = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
    y = label_matrix(labels.loc[common].reset_index())
    score = lambda names: float(emd_per_sample(y, pointwise_median([oofs[n].loc[common, TARGET_COLUMNS].to_numpy() for n in names])).mean())

    for n in oofs:
        print(f"  {n:30s} OOF EMD {score([n]):.3f}")
    if args.all:
        chosen = list(oofs)
    else:
        chosen, best = [], np.inf
        while len(chosen) < args.max_members:
            cand = [(score(chosen + [n]), n) for n in oofs]  # repeats allowed = weighting
            s, n = min(cand)
            if s >= best - 1e-6:
                break
            chosen.append(n)
            best = s
    final = score(chosen)
    print(f"ensemble {chosen} -> OOF EMD {final:.3f} on {len(common)} samples")
    pred = make_valid(pointwise_median([tests[n][TARGET_COLUMNS].to_numpy() for n in chosen]))
    write_submission(test_ids, pred, args.out)
    meta = {"members": chosen, "oof_emd": final, "n_oof": len(common)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
