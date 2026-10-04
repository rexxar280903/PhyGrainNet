"""Shrink the ETS photogranulometry images (425 GB, 8192x5464 @ 39.4 um/px) to the
canonical PPM and write a catalog, so they fit in a Kaggle Dataset.

    python scripts/prepare_ets.py --src /data/photogranulometry/soil_image \
        --out /data/ets_canonical --target-ppm 10 --max-per-sample 6

Run it where the raw data lives (Globus download), then upload --out as a
private Kaggle Dataset. The file-name pattern is not documented precisely, so
check `--inspect` first and adjust --id-regex if needed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from PIL import Image

from phygrainnet.constants import IMAGE_EXTENSIONS
from phygrainnet.data.imaging import load_rgb, normalize_color, to_ppm

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-ppm", type=float, default=10.0)
    ap.add_argument("--source-um-per-px", type=float, default=39.4)
    ap.add_argument("--id-regex", default=r"^(?P<sample>[A-Za-z]*\d+)")
    ap.add_argument("--max-per-sample", type=int, default=6)
    ap.add_argument("--moisture", choices=["any", "dry", "humid"], default="any")
    ap.add_argument("--color-mode", default="none", help="colour normalisation is applied again at load time")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    rx = re.compile(args.id_regex)
    rows = []
    for p in files:
        m = rx.search(p.stem)
        low = p.stem.lower()
        moisture = "dry" if "dry" in low or "sec" in low else ("humid" if "humid" in low or "hum" in low else "unknown")
        rows.append({"src": str(p), "file": p.name, "sample_id": m.group("sample") if m else None, "moisture": moisture})
    df = pd.DataFrame(rows)
    if args.inspect:
        print(df.head(30).to_string())
        print(df["moisture"].value_counts())
        print(f"{df['sample_id'].nunique()} sample ids parsed, {df['sample_id'].isna().sum()} unparsed")
        return
    if args.moisture != "any":
        df = df[df["moisture"] == args.moisture]
    df = df.dropna(subset=["sample_id"]).groupby("sample_id").head(args.max_per_sample)

    src_ppm = 1000.0 / args.source_um_per_px
    args.out.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in df.itertuples():
        dst = args.out / "images" / f"{r.sample_id}__{Path(r.file).stem}.jpg"
        if not dst.exists():
            img = normalize_color(to_ppm(load_rgb(r.src), src_ppm, args.target_ppm), args.color_mode)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(dst, quality=92)
        out_rows.append({"path": str(dst.relative_to(args.out)), "file": dst.name, "sample_id": r.sample_id, "moisture": r.moisture,
                         "camera": "ets_canon_r5", "ppm": args.target_ppm, "split": "external"})
        print(dst.name, flush=True)
    pd.DataFrame(out_rows).to_csv(args.out / "ets_catalog.csv", index=False)
    print(f"wrote {len(out_rows)} images, catalog -> {args.out / 'ets_catalog.csv'}")
    print("paths are relative to the catalog file, so the folder can be uploaded to Kaggle as-is")


if __name__ == "__main__":
    main()
