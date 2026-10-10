import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from phygrainnet.constants import TARGET_COLUMNS

REPO = Path(__file__).resolve().parents[1]


def run_script(name, *args, success=True):
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / name), *map(str, args)],
                          cwd=REPO, env=env, capture_output=True, text=True)
    assert (proc.returncode == 0) == success, proc.stdout + proc.stderr
    return proc


def test_baselines_audit_and_verified_backup(fake_root, tmp_path):
    audit = tmp_path / "phygrainnet" / "audit"
    run_script("audit_dataset.py", "--data-root", fake_root, "--out-dir", audit)
    assert json.loads((audit / "dataset_audit.json").read_text())["passed"]
    out = tmp_path / "outputs"
    run_script("baseline_prior.py", "--data-root", fake_root, "--out-dir", out,
               "--registry", tmp_path / "registry.csv", "--out", tmp_path / "submission_A000.csv")
    for name in ("mean", "median"):
        folder = out / f"A000_{name}"
        assert {"oof.csv", "test.csv", "targets.csv", "groups.csv", "metrics.json", "config.json", "environment.json"} <= {p.name for p in folder.iterdir()}
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["cv_complete"] and len(metrics["fold_splits"]) == 5
    archive = tmp_path / "results.zip"
    run_script("export_results.py", "--work-root", tmp_path, "--out", archive)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        for name, info in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == info["sha256"]
        assert "outputs/A000_median/targets.csv" in manifest["files"]
        assert "phygrainnet/audit/dataset_audit.json" in manifest["files"]
        assert "source/scripts/predict.py" in manifest["files"]


def test_saved_checkpoint_restores_predictions_and_smoke_is_rejected(tiny_cfg, tmp_path):
    from phygrainnet.training.pipelines import run_cnn

    cfg = copy.deepcopy(tiny_cfg)
    cfg["experiment_id"] = "C100_smoke"
    cfg["cv"]["max_folds"] = 1
    metrics = run_cnn(cfg)
    assert metrics["cv_complete"] is False
    assert pd.read_csv(cfg["output"]["registry"]).iloc[0]["status"] == "smoke_only"
    folder = Path(cfg["output"]["dir"]) / "C100_smoke"
    restored = tmp_path / "restored.csv"
    run_script("predict.py", "--checkpoints", folder / "fold0.pt", "--data-root", cfg["data"]["root"],
               "--cache-dir", tmp_path / "restore_cache", "--out", restored)
    expected = pd.read_csv(folder / "test.csv")
    actual = pd.read_csv(restored)
    assert actual.sample_id.tolist() == expected.sample_id.tolist()
    np.testing.assert_allclose(actual[TARGET_COLUMNS], expected[TARGET_COLUMNS], atol=5.1e-5, rtol=0)
    proc = run_script("ensemble.py", "--runs", folder, "--data-root", cfg["data"]["root"],
                      "--out", tmp_path / "invalid_ensemble.csv", success=False)
    assert "incomplete CV" in proc.stderr
    assert not (tmp_path / "invalid_ensemble.csv").exists()


def test_cache_changes_when_source_or_physical_scale_changes(fake_root, tmp_path):
    from phygrainnet.data.imaging import cache_path

    images = list(fake_root.rglob("*.jpg"))
    a = cache_path(tmp_path, images[0], 2, 0, "none", ppm=2)
    b = cache_path(tmp_path, images[0], 2, 0, "none", ppm=4)
    assert a != b
    alternate = tmp_path / images[0].name
    alternate.write_bytes(images[0].read_bytes())
    assert cache_path(tmp_path, alternate, 2, 0, "none", ppm=2) != a


def test_audit_rejects_invalid_labels_and_cross_split_duplicates(fake_root, tmp_path):
    import shutil

    root = tmp_path / "data"
    shutil.copytree(fake_root, root)
    labels = root / "Training_labels_updated.csv"
    table = pd.read_csv(labels)
    table.iloc[0, 1] = np.nan
    table.to_csv(labels, index=False)
    train_image = next(root.rglob("Motorola_Edge_F827_01.jpg"))
    test_image = next(root.rglob("iPhone14*JPG"))
    test_image.write_bytes(train_image.read_bytes())
    audit = tmp_path / "audit"
    run_script("audit_dataset.py", "--data-root", root, "--out-dir", audit, success=False)
    report = json.loads((audit / "dataset_audit.json").read_text())
    assert report["passed"] is False
    assert "invalid training labels" in report["fatal_problems"]
    assert "identical image bytes across different samples or splits" in report["fatal_problems"]


def test_runner_cells_compile_and_external_stages_are_disabled():
    nb = json.loads((REPO / "notebooks" / "kaggle_runner.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "kaggle_runner", "exec")
    source = "".join(nb["cells"][1]["source"])
    assert "RUN_ETS = False" in source and "RUN_FULL_CV = False" in source


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_cannot_be_disguised_as_valid_submission(bad):
    from phygrainnet.postprocess import make_valid

    prediction = np.linspace(0, 100, 11)[None]
    prediction[0, 3] = bad
    with pytest.raises(ValueError, match="non-finite predictions"):
        make_valid(prediction)
