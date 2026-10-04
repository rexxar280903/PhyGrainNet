"""Convert the ETS photogranulometry PSD Excel file to the 11 competition points.

Usage (Kaggle or local):

    python scripts/convert_ets_labels.py \
        --input /kaggle/input/ets-photogranulometry/psd.xlsx \
        --output /kaggle/working/phygrainnet/ets_labels_11pt.csv

    # inspect the raw file first if the columns are not detected:
    python scripts/convert_ets_labels.py --input psd.xlsx --inspect

    # estimate how much accuracy the interpolation itself costs, using the
    # competition training labels and the detected ETS sieve set:
    python scripts/convert_ets_labels.py --input psd.xlsx --output out.csv \
        --roundtrip-labels /kaggle/input/soil-grain-size-from-photos/Training_labels_updated.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from phygrainnet.data.ets_labels import (
    TARGET_COLUMNS,
    convert_dataframe,
    load_table,
    roundtrip_error,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--sheet", default=0, help="sheet name or index (Excel only)")
    ap.add_argument("--header-row", type=int, default=0)
    ap.add_argument("--id-column", default=None)
    ap.add_argument("--default-unit", default="mm", choices=["mm", "um"],
                    help="unit for headers that are bare numbers")
    ap.add_argument("--inspect", action="store_true", help="print sheets/columns/head and exit")
    ap.add_argument("--roundtrip-labels", type=Path, default=None)
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    if args.inspect:
        if args.input.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            print("sheets:", pd.ExcelFile(args.input).sheet_names)
        df = load_table(args.input, sheet, args.header_row)
        print("columns:", list(df.columns))
        print(df.head(10).to_string())
        return

    if args.output is None:
        ap.error("--output is required unless --inspect is used")

    df = load_table(args.input, sheet, args.header_row)
    out, report = convert_dataframe(df, id_column=args.id_column, default_unit=args.default_unit)
    print(report)

    print("\nlabel coverage per competition point (share of samples):")
    for c in TARGET_COLUMNS:
        print(f"  {c:>7} mm : {out[f'mask_{c}'].mean():.2f}")
    if "passing_finest_sieve" in out:
        q = out["passing_finest_sieve"].describe(percentiles=[0.1, 0.5, 0.9])
        print("\n% passing finest sieve (fines upper bound):")
        print(q.round(2).to_string())

    if args.roundtrip_labels is not None:
        lab = pd.read_csv(args.roundtrip_labels)
        y = lab[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        rt = roundtrip_error(y, np.array(report.sieves_mm))
        print("\ninterpolation round-trip EMD on competition labels (supported points only):")
        print(json.dumps({k: round(v, 3) for k, v in rt.items()}, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nwrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
