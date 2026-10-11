"""Inspect actual model sizes; optionally calibrate training time on the device."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from phygrainnet.config import load_config
from phygrainnet.profiling import benchmark, hardware_info, model_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phygrainnet/kaggle_scratch.yaml")
    ap.add_argument("--out", type=Path, default=Path("outputs/profile.json"))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    result = {"config": cfg, "hardware": hardware_info(), "model": model_profile(cfg)}
    if args.benchmark:
        result["benchmark"] = benchmark(cfg, args.device, args.warmup, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "parameters": result["model"]["parameters"],
                      "forward_gmacs_batch": result["model"]["forward_macs_batch"] / 1e9,
                      "benchmark": result.get("benchmark")}, indent=2))


if __name__ == "__main__":
    main()
