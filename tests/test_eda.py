import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phygrainnet.eda import d_value, label_table, run_eda
from phygrainnet.data.labels import load_train_labels, site_groups

ROOT = Path(__file__).resolve().parents[1]


def test_d_value_log_interpolation():
    curve = np.array([0, 0, 0, 0, 10, 30, 50, 70, 90, 100, 100], float)
    assert np.isclose(d_value(curve, 50), 2.0)
    assert np.isclose(d_value(curve, 10), 0.2)
    # halfway between 0.63 and 2 mm in log space
    assert np.isclose(np.log10(d_value(curve, 40)), (np.log10(0.63) + np.log10(2)) / 2)
    assert np.isnan(d_value(np.full(11, 100.0), 10))  # everything finer than 2 µm


def test_label_table_fractions(fake_root):
    labels = load_train_labels(fake_root)
    t = label_table(labels, site_groups(labels.sample_id.tolist()))
    total = t[["clay", "silt", "sand", "gravel", "cobbles"]].sum(axis=1)
    assert np.allclose(total, 100, atol=1e-3)
    assert set(t.family) <= {"fine", "gravelly"}
    assert t.set_index("sample_id").loc["H374", "family"] == "gravelly"
    assert t.set_index("sample_id").loc["F827", "family"] == "fine"


def test_run_eda_writes_everything(tiny_cfg, tmp_path):
    cfg = copy.deepcopy(tiny_cfg)
    out = tmp_path / "eda"
    s = run_eda(cfg, out, with_features=True, n_jobs=1)
    for f in s["figures"]:
        assert (out / f).stat().st_size > 10_000
    assert len(s["figures"]) == 6
    assert s["n_train"] == 8 and s["n_test"] == 3
    assert (out / "eda_report.md").read_text().startswith("# EDA summary")
    nn = pd.read_csv(out / "test_nearest_train.csv")
    assert len(nn) == 3
    # the feature table is cached where A001 will look for it
    from phygrainnet.training.pipelines import feature_cache_path

    assert feature_cache_path(cfg).exists()
    json.loads((out / "eda_summary.json").read_text())


def test_make_submission_picks_lowest_cv(tiny_cfg, fake_root, tmp_path):
    from phygrainnet.training.pipelines import run_classical

    cfg = copy.deepcopy(tiny_cfg)
    cfg["cv"]["strategy"] = "logo"
    cfg["experiment_id"] = "A001"
    cfg["classical"] = {"models": [{"name": "ridge", "kind": "ridge_clr", "alpha": 10},
                                   {"name": "prior", "kind": "prior_median"}]}
    res = run_classical(cfg)
    best = min(res, key=lambda k: res[k]["cv_emd"])
    out_dir = Path(cfg["output"]["dir"])
    sub = tmp_path / "submission.csv"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--runs",
                        str(out_dir / "A001_ridge"), str(out_dir / "A001_prior"), "--pick-best",
                        "--data-root", str(fake_root), "--out", str(sub)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    choice = json.loads(sub.with_suffix(".json").read_text())
    assert choice["experiment_id"] == best
    df = pd.read_csv(sub)
    assert len(df) == 3 and (df["200"] == 100).all()
