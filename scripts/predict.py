"""Restore saved fold weights and produce a submission without training.

    python scripts/predict.py --checkpoints /kaggle/input/previous-run/outputs/C100/fold*.pt \
        --out /kaggle/working/submission_C100_restored.csv
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import _bootstrap  # noqa: F401

from phygrainnet.models.phygrainnet import build_model
from phygrainnet.postprocess import pointwise_median
from phygrainnet.submission import write_submission
from phygrainnet.training.data import records_from_catalog
from phygrainnet.training.pipelines import Competition
from phygrainnet.training.trainer import device_auto, load_checkpoint, predict_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/cache"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = device_auto()
    predictions = []
    for path in args.checkpoints:
        ckpt = load_checkpoint(path)
        if ckpt.get("encoders_only"):
            raise ValueError(f"{path}: encoder pretraining weights cannot produce a submission")
        cfg = copy.deepcopy(ckpt["config"])
        cfg["data"].update(root=str(args.data_root), cache_dir=str(args.cache_dir))
        comp = Competition(cfg)
        d = cfg["data"]
        records = records_from_catalog(comp.catalog[comp.catalog.split == "test"], None,
            float(d["target_ppm"]), float(d.get("border_frac", 0.06)), d.get("color_mode", "grayworld"),
            d.get("cache_dir"), sample_ids=comp.test_ids)
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        predictions.append(predict_records(cfg, model, records, device))
        del model
        print(f"restored {path}", flush=True)
    write_submission(comp.test_ids, pointwise_median(predictions), args.out)
    print(f"wrote validated submission -> {args.out}")


if __name__ == "__main__":
    main()
