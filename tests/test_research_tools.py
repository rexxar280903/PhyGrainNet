import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phygrainnet.constants import DIAMETERS_MM
from phygrainnet.profiling import benchmark, model_profile
from phygrainnet.reporting import cluster_interval, compare_runs, diameter_at, fractions, load_run, summarize_run


def test_fraction_boundaries_and_censored_quantiles():
    c = np.array([20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100], float)
    np.testing.assert_allclose(fractions(c[None]), [[20, 30, 30, 20]])
    assert diameter_at(c, 10) is None
    assert diameter_at(c, 20) == 0.002
    assert diameter_at(c, 50) == 0.063
    assert diameter_at(c, 55) == pytest.approx(np.sqrt(0.063 * 0.2))
    plateau = c.copy()
    plateau[4] = 50
    assert diameter_at(plateau, 50) == 0.063
    with pytest.raises(ValueError):
        fractions(np.zeros((2, 11)))


def test_cluster_interval_weights_samples_and_is_reproducible():
    a = cluster_interval([0, 0, 10], [1, 1, 2], repeats=200, seed=42)
    assert a["estimate"] == pytest.approx(10 / 3)
    assert a["lower"] == 0 and a["upper"] == 10
    assert a == cluster_interval([0, 0, 10], [1, 1, 2], repeats=200, seed=42)
    assert a["clusters"] == 2
    with pytest.raises(ValueError):
        cluster_interval([float("nan")], [1])


def test_synthetic_profile_and_single_tile_gradient(tiny_cfg):
    from phygrainnet.models.phygrainnet import build_model
    cfg = copy.deepcopy(tiny_cfg)
    cfg["training"]["tiles_per_sample"] = 1
    model = build_model(cfg)
    profile = model_profile(cfg)
    assert profile["parameters"] == sum(p.numel() for p in model.parameters())
    assert profile["forward_macs_batch"] == sum(r["macs"] for r in profile["layers"])
    x = torch.randn(2, 1, 2, 3, 32, 32)
    model(x)["cumulative"][:, :10].mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="valid tile"):
        model(x, torch.zeros(2, 1))
    measured = benchmark(cfg, device="cpu", warmup=0, steps=1)
    assert measured["seconds_per_step"] > 0
    assert measured["peak_allocated_mib"] is None


def test_complete_report_and_rejection_of_corrupt_provenance(tiny_cfg):
    from phygrainnet.training.pipelines import run_cnn
    cfg = copy.deepcopy(tiny_cfg)
    cfg["experiment_id"] = "TEST_REPORT"
    run_cnn(cfg)
    path = Path(cfg["output"]["dir"]) / "TEST_REPORT"
    run = load_run(path)
    summary = summarize_run(run, repeats=50)
    assert summary["n_oof"] == 8
    assert len(summary["curves"]) == 8
    assert sum(fractions(run["p"])[0]) == pytest.approx(100)
    comparison = compare_runs(run, run, repeats=50)
    assert comparison["delta_emd_ci95"]["estimate"] == 0
    changed = copy.deepcopy(run)
    changed["folds"][0]["val_ids"] = []
    with pytest.raises(ValueError, match="fold splits"):
        compare_runs(changed, run)
    metrics = json.loads((path / "metrics.json").read_text())
    metrics["fold_splits"][0]["train_ids"].append(metrics["fold_splits"][0]["val_ids"][0])
    (path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="fold coverage"):
        load_run(path)
    metrics["cv_complete"] = False
    (path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="incomplete CV"):
        load_run(path)


def test_ablation_settings_reach_model_and_data(tiny_cfg):
    from phygrainnet.config import load_config
    from phygrainnet.models.phygrainnet import build_model
    root = Path(__file__).resolve().parents[1]
    configs = [load_config(p) for p in (root / "configs/ablations").glob("*.yaml")]
    assert len(configs) == 7
    for cfg in configs:
        model = build_model(cfg)
        assert cfg["cv"]["n_folds"] == 5
        assert cfg["training"]["epochs"] == 40
        if cfg["experiment_id"] == "F101":
            assert model.encoders[0].texture is None
        if cfg["experiment_id"] == "F107":
            assert model.pool.use_attention is False
        if cfg["experiment_id"] == "F106":
            assert cfg["data"]["canonical_scale"] is False
        if cfg["experiment_id"] == "F103":
            assert cfg["inference"]["max_tiles_per_sample"] == 1


def test_source_package_excludes_secrets_and_verifies_checksums(tmp_path):
    import hashlib
    import importlib.util
    import zipfile
    script = Path(__file__).resolve().parents[1] / "scripts/export_source.py"
    spec = importlib.util.spec_from_file_location("export_source", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/a.py").write_text("print('hello')")
    (repo / "src/secret.pem").write_text("not a real key")
    (repo / ".env").write_text("not real credentials")
    archive = tmp_path / "source.zip"
    manifest = mod.export_source(repo, archive)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {"src/a.py", "source_manifest.json"}
        assert hashlib.sha256(z.read("src/a.py")).hexdigest() == manifest["files"]["src/a.py"]["sha256"]
