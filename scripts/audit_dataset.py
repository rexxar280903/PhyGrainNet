from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the live Kaggle competition dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/soil-grain-size-from-photos"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/phygrainnet/dataset_audit.json"),
    )
    args = parser.parse_args()

    root = args.data_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    files = discover_files(root)
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    csvs = [p for p in files if p.suffix.lower() == ".csv"]

    dims = Counter()
    unreadable = []
    for p in images:
        try:
            with Image.open(p) as im:
                dims[f"{im.width}x{im.height}"] += 1
        except Exception as exc:
            unreadable.append({"path": str(p), "error": repr(exc)})

    csv_summary = {}
    for p in csvs:
        try:
            df = pd.read_csv(p)
            csv_summary[str(p.relative_to(root))] = {
                "rows": int(len(df)),
                "columns": list(map(str, df.columns)),
                "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
            }
        except Exception as exc:
            csv_summary[str(p.relative_to(root))] = {"error": repr(exc)}

    report = {
        "data_root": str(root),
        "total_files": len(files),
        "image_count": len(images),
        "csv_count": len(csvs),
        "image_dimensions": dict(dims),
        "csvs": csv_summary,
        "unreadable_images": unreadable,
        "next_manual_checks": [
            "identify the physical sample key",
            "verify photos per physical sample",
            "verify PPM metadata key and range",
            "verify target column order and monotonicity",
            "check duplicate and near-duplicate risks",
            "document train/test leakage risks",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
