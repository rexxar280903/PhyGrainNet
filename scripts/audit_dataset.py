"""P2 dataset audit: index photos, verify labels, report scale per camera and CV groups.

    python scripts/audit_dataset.py --data-root /kaggle/input/soil-grain-size-from-photos
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from phygrainnet.constants import TARGET_COLUMNS
from phygrainnet.data.catalog import build_catalog, catalog_problems
from phygrainnet.data.labels import label_matrix, load_test_ids, load_train_labels, site_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/phygrainnet/audit"))
    ap.add_argument("--scale-mode", default="resize", choices=["resize", "native"])
    args = ap.parse_args()

    labels = load_train_labels(args.data_root)
    test_ids = load_test_ids(args.data_root)
    y = label_matrix(labels)
    cat = build_catalog(args.data_root, scale_mode=args.scale_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cat.to_csv(args.out_dir / "catalog.csv", index=False)

    groups = site_groups(labels["sample_id"].tolist())
    group_map = {}
    for sid, g in zip(labels["sample_id"], groups):
        group_map.setdefault(int(g), []).append(sid)

    report = {
        "n_train_samples": len(labels),
        "n_test_samples": len(test_ids),
        "n_photos": {s: int((cat.split == s).sum()) for s in ("train", "test")},
        "photos_per_sample": cat.groupby(["split", "sample_id"]).size().groupby("split").describe().round(2).to_dict(),
        "camera_by_split": cat.groupby(["split", "camera"]).size().unstack(fill_value=0).to_dict(),
        "image_sizes": cat.groupby(["camera", "width", "height"]).size().reset_index(name="n").to_dict("records"),
        "ppm_by_camera": cat.groupby("camera")["ppm"].agg(["min", "max"]).round(3).to_dict("index"),
        "field_of_view_mm": cat.groupby("camera")[["field_w_mm", "field_h_mm"]].median().round(1).to_dict("index"),
        "labels": {
            "non_finite": int((~np.isfinite(y)).sum()),
            "monotone_violations": int((np.diff(y, axis=1) < -1e-9).sum()),
            "out_of_range": int(((y < 0) | (y > 100)).sum()),
            "last_not_100": labels.loc[y[:, -1] != 100, "sample_id"].tolist(),
            "column_means": dict(zip(TARGET_COLUMNS, y.mean(0).round(2).tolist())),
        },
        "cv_groups": group_map,
        "problems": catalog_problems(cat, labels["sample_id"].tolist(), test_ids),
    }
    by_hash = {}
    for row in cat.itertuples():
        digest = hashlib.sha256(Path(row.path).read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append({"file": row.file, "split": row.split, "sample_id": row.sample_id})
    report["exact_duplicates"] = [rows for rows in by_hash.values() if len(rows) > 1]
    fatal = [p for p in report["problems"] if "without " in p or "no photos" in p]
    if any(report["labels"][k] for k in ("non_finite", "monotone_violations", "out_of_range", "last_not_100")):
        fatal.append("invalid training labels")
    if labels.sample_id.duplicated().any() or len(set(test_ids)) != len(test_ids):
        fatal.append("duplicate sample IDs")
    if set(labels.sample_id) & set(test_ids):
        fatal.append("train/test sample overlap")
    if "ppm" not in cat or not np.isfinite(cat.ppm).all() or (cat.ppm <= 0).any():
        fatal.append("invalid physical scale")
    if any(len({(r['split'], r['sample_id']) for r in rows}) > 1 for rows in report["exact_duplicates"]):
        fatal.append("identical image bytes across different samples or splits")
    report["fatal_problems"] = fatal
    report["passed"] = not fatal
    (args.out_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    if report["problems"]:
        print("\n!! fix the problems above before training (see docs/DATASET.md)")
    if fatal:
        raise SystemExit("dataset audit failed: " + "; ".join(fatal))


if __name__ == "__main__":
    main()
