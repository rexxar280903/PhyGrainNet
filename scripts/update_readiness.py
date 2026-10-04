"""Regenerate the README status block from project_status.json.

    python scripts/update_readiness.py          # rewrite README.md in place
    python scripts/update_readiness.py --check  # exit 1 if README is stale (CI)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- STATUS:START -->", "<!-- STATUS:END -->"


def bar(pct: float, width: int = 26) -> str:
    n = int(round(pct / 100 * width))
    return "█" * n + "░" * (width - n)


def render(s: dict) -> str:
    cv = s.get("best_local_cv_emd")
    lb = s.get("best_public_score")
    lines = [
        START,
        f"**Overall project progress: {s['overall_progress_percent']}%**  ",
        f"`{bar(s['overall_progress_percent'])} {s['overall_progress_percent']}%`",
        "",
        f"**Competition readiness: {s['competition_readiness_percent']}%**  ",
        f"`{bar(s['competition_readiness_percent'])} {s['competition_readiness_percent']}%`",
        "",
        f"**Current phase:** {s['current_phase']} — {s['current_phase_name']}  ",
        f"**Next phase:** {s['next_phase']} — {s['next_phase_name']}  ",
        f"**Best grouped-CV EMD:** {cv if cv is not None else 'not established yet'}  ",
        f"**Best public LB (not used for selection):** {lb if lb is not None else 'no submission yet'}  ",
        f"**Deadline:** {s.get('deadline', '2026-11-30 18:00 WIB')}",
        END,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    status = json.loads((ROOT / "project_status.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        sys.exit("README.md lacks status markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    new = head + render(status) + tail
    if args.check:
        sys.exit(0 if new == readme else "README status block is stale: run scripts/update_readiness.py")
    (ROOT / "README.md").write_text(new, encoding="utf-8")
    print("README status block updated")


if __name__ == "__main__":
    main()
