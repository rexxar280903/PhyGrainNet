"""Turn one run's test.csv into a validated Kaggle submission.

    python scripts/make_submission.py --run outputs/A001_ridge --out submission_A001.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from phygrainnet.constants import TARGET_COLUMNS
from phygrainnet.data.labels import load_test_ids
from phygrainnet.submission import read_prediction_csv, write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    test_ids = load_test_ids(args.data_root)
    df = read_prediction_csv(args.run / "test.csv").set_index("sample_id").loc[test_ids]
    write_submission(test_ids, df[TARGET_COLUMNS].to_numpy(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
