"""Package the current local source for Kaggle; excludes raw data and secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def export_source(repo: Path, out: Path) -> dict:
    files = {}
    for folder in ["src", "scripts", "configs", "notebooks", "web", "docs", "tests"]:
        for path in (repo / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() in {".py", ".yaml", ".json", ".md", ".html", ".css", ".js", ".cjs", ".ipynb", ".jpg"}:
                files[path.relative_to(repo).as_posix()] = path
    for name in ["README.md", "requirements.txt", "pyproject.toml", "project_status.json", "LICENSE"]:
        if (repo / name).exists():
            files[name] = repo / name
    if not files or out.resolve() in [p.resolve() for p in files.values()]:
        raise ValueError("source files required; archive must be outside source inputs")
    manifest = {"schema_version": 1, "purpose": "PhyGrainNet local source bundle for Kaggle", "files": {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            z.writestr(name, data)
        z.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(out) as z:
        for name, info in manifest["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != info["sha256"]:
                raise ValueError("source archive checksum failure")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/phygrainnet_source.zip"))
    args = ap.parse_args()
    manifest = export_source(Path(__file__).resolve().parents[1], args.out)
    print(f"Verified source bundle: {args.out}; {len(manifest['files'])} files; no raw data/venv/git/secrets")


if __name__ == "__main__":
    main()
