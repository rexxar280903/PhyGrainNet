"""Export validated real OOF evidence for dashboard import and paper tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from phygrainnet.reporting import compare_runs, load_run, summarize_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/research_report.json"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    runs = [load_run(p) for p in args.runs]
    result = {"schema_version": 1, "evidence_type": "complete_grouped_oof",
              "runs": [summarize_run(r, args.bootstrap) for r in runs],
              "comparisons": [compare_runs(r, runs[0], args.bootstrap) for r in runs[1:]]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Validated {len(runs)} runs; reference={runs[0]['id']}; exported {args.out}")


if __name__ == "__main__":
    main()
