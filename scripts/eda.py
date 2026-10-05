"""P3 EDA: label curves, physical-scale thumbnails, camera/colour shift, feature space.

    python scripts/eda.py                                  # Kaggle defaults
    python scripts/eda.py --no-features                    # skip the ~CPU-heavy feature pass
    python scripts/eda.py --config configs/classical/features_v1.yaml data.color_mode=lab

Writes PNGs, CSV tables, eda_summary.json and eda_report.md to --out-dir. The per-photo
feature table is cached where A001 looks for it, so A001 afterwards starts instantly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from phygrainnet.config import load_config
from phygrainnet.eda import run_eda

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "classical" / "features_v1.yaml"))
    ap.add_argument("--data-root", default=None, help="overrides data.root")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/phygrainnet/eda"))
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.n_jobs=4")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    s = run_eda(cfg, args.out_dir, with_features=not args.no_features)
    print((args.out_dir / "eda_report.md").read_text(encoding="utf-8"))
    print(json.dumps({k: s[k] for k in ("baseline_logo_emd", "families", "figures")}, indent=2))


if __name__ == "__main__":
    main()
