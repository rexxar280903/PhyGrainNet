"""Single entry point for every pipeline; the config's `pipeline:` key picks it.

    python scripts/train.py --config configs/classical/features_v1.yaml
    python scripts/train.py --config configs/phygrainnet/mv_scratch.yaml training.epochs=2
    python scripts/train.py --config configs/pretrain/ssl.yaml
    python scripts/train.py --config configs/pretrain/ets.yaml
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from phygrainnet.config import load_config
from phygrainnet.training import pipelines
from phygrainnet.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. training.epochs=3")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.get("seed", 42)))
    kind = cfg.get("pipeline", "cnn")
    print(f"[train] pipeline={kind} experiment={cfg.get('experiment_id')}")
    if kind == "classical":
        res = pipelines.run_classical(cfg, args.config)
        print(json.dumps({k: round(v["cv_emd"], 3) for k, v in res.items()}, indent=2))
    elif kind == "cnn":
        pipelines.run_cnn(cfg, args.config)
    elif kind == "ssl":
        pipelines.run_ssl(cfg)
    elif kind == "ets_pretrain":
        pipelines.run_ets_pretrain(cfg)
    else:
        raise SystemExit(f"unknown pipeline {kind!r}")


if __name__ == "__main__":
    main()
