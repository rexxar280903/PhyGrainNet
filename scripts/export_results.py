"""Bundle run artifacts, audit, registry and source snapshot with SHA-256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def export_results(work_root: Path, repo: Path, out: Path) -> dict:
    files = {}
    for name in ("outputs", "phygrainnet/audit"):
        folder = work_root / name
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file():
                    files[str(path.relative_to(work_root)).replace("\\", "/")] = path
    for path in work_root.glob("submission*"):
        if path.is_file():
            files[path.name] = path
    if (work_root / "registry.csv").exists():
        files["registry.csv"] = work_root / "registry.csv"
    for folder in ("src", "scripts", "configs", "notebooks"):
        for path in (repo / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files["source/" + str(path.relative_to(repo)).replace("\\", "/")] = path
    for name in ("README.md", "requirements.txt", "pyproject.toml", "project_status.json"):
        if (repo / name).exists():
            files["source/" + name] = repo / name
    if not any(name.startswith("outputs/") for name in files):
        raise ValueError("no run outputs found; nothing to archive")
    if out.resolve() in {p.resolve() for p in files.values()}:
        raise ValueError("archive must be outside the archived inputs")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    manifest = {"git_commit": commit, "working_tree_dirty": dirty, "files": {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            digest = hashlib.sha256()
            with z.open(name, "w", force_zip64=True) as dest, path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    dest.write(block)
            manifest["files"][name] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError("archive integrity check failed")
    temp.replace(out)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/phygrainnet_results.zip"))
    args = ap.parse_args()
    manifest = export_results(args.work_root, Path(__file__).resolve().parents[1], args.out)
    print(f"verified archive: {args.out} ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()
