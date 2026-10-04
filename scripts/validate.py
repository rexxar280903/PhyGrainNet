"""Check a submission CSV against the scorer's structural rules before uploading.

    python scripts/validate.py submission.csv --data-root /kaggle/input/soil-grain-size-from-photos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from phygrainnet.data.labels import load_test_ids
from phygrainnet.submission import SubmissionError, validate_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    try:
        validate_submission(df, load_test_ids(args.data_root))
    except SubmissionError as exc:
        raise SystemExit(f"INVALID: {exc}")
    print(f"OK: {args.csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
