"""Turn a run's test.csv into a validated Kaggle submission.

    python scripts/make_submission.py --run outputs/A001_ridge10 --out submission_A001.csv
    python scripts/make_submission.py --runs outputs/A001_* --pick-best --out submission.csv

With --pick-best the run with the lowest grouped-CV EMD (metrics.json) is used — never
the public leaderboard (DEC-006). A sidecar <out>.json records the choice for
docs/LEADERBOARD_LOG.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from phygrainnet.constants import TARGET_COLUMNS
from phygrainnet.data.labels import load_test_ids
from phygrainnet.submission import read_prediction_csv, write_submission


def rank_runs(runs: list[Path]) -> list[dict]:
    rows = []
    for r in runs:
        mf = r / "metrics.json"
        if not (mf.exists() and (r / "test.csv").exists()):
            print(f"  skip {r} (no metrics.json/test.csv)")
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        rows.append({"run": str(r), "experiment_id": m.get("experiment_id", r.name), "cv_emd": float(m["cv_emd"]),
                     "cv_emd_std_over_groups": float(m.get("cv_emd_std_over_groups", float("nan"))),
                     "n_oof": len(m.get("per_sample", {})), "git_commit": m.get("git_commit", "unknown")})
    return sorted(rows, key=lambda d: d["cv_emd"])


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path)
    src.add_argument("--runs", nargs="+", type=Path)
    ap.add_argument("--pick-best", action="store_true", help="with --runs: lowest grouped-CV EMD wins")
    ap.add_argument("--min-oof", type=int, default=0, help="ignore runs scored on fewer OOF samples (quick runs)")
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/soil-grain-size-from-photos"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    chosen = {"run": str(args.run)}
    if args.runs:
        if not args.pick_best and len(args.runs) > 1:
            raise SystemExit("several --runs given: add --pick-best (or use scripts/ensemble.py)")
        ranked = [r for r in rank_runs(args.runs) if r["n_oof"] >= args.min_oof]
        if not ranked:
            raise SystemExit("no usable run")
        for r in ranked:
            print(f"  {r['experiment_id']:24s} CV EMD {r['cv_emd']:8.3f}  (group std {r['cv_emd_std_over_groups']:.2f}, n={r['n_oof']})")
        chosen = ranked[0]
        print(f"best by grouped CV: {chosen['experiment_id']} ({chosen['cv_emd']:.3f})")
        args.run = Path(chosen["run"])

    test_ids = load_test_ids(args.data_root)
    df = read_prediction_csv(args.run / "test.csv").set_index("sample_id").loc[test_ids]
    write_submission(test_ids, df[TARGET_COLUMNS].to_numpy(), args.out)
    args.out.with_suffix(".json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
