import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phygrainnet.config import apply_override, deep_merge, load_config
from phygrainnet.data.catalog import build_catalog, catalog_problems, norm_key
from phygrainnet.data.labels import grouped_kfold, load_train_labels, site_groups
from phygrainnet.features.granulometry import image_features
from phygrainnet.models.classical import CLRRegressor, KNNMedian, clr_to_cdf, cdf_to_clr
from phygrainnet.models.phygrainnet import PhyGrainNetMV
from phygrainnet.postprocess import make_valid, pointwise_median
from phygrainnet.submission import SubmissionError, build_submission, validate_submission, write_submission

from conftest import REAL_LABELS, TEST_IDS

REAL_TRAIN_IDS = ["F827", "G190", "H030", "H037", "H038", "H126", "H181", "H183", "H366", "H367", "H368", "H371",
                  "H372", "H374", "H405", "H493", "H516", "H549", "H615", "H616", "H617", "H637", "H666", "H668"]


def test_norm_key_matches_umlaut_and_comma():
    assert norm_key("HPC_Münster_BS6_9,0-10m") == norm_key("HPC_Muenster_BS6_9_0-10m")


def test_catalog_parses_real_naming(fake_root):
    cat = build_catalog(fake_root)
    assert cat["sample_id"].notna().all() and cat["camera"].notna().all()
    assert catalog_problems(cat, list(REAL_LABELS), TEST_IDS) == []
    row = cat[cat.file.str.startswith("Motorola_Edge_60_fusion")].iloc[0]
    assert row.camera == "Motorola Edge 60 Fusion" and row.sample_id == "H374"
    assert set(cat.loc[cat.split == "test", "sample_id"]) == set(TEST_IDS)
    kk = cat[cat.file == "iPhone16_HPC_Kleinkummerfeld 2-2 (3).JPG"].iloc[0]
    assert kk.sample_id == "HPC_Kleinkummerfeld 2-2" and kk.view == 3
    assert cat.loc[cat.camera == "Samsung A52", "sample_id"].tolist() == ["H405", "H405"]
    assert np.allclose(cat["resize_factor"], 1.0)


def test_site_groups_on_real_ids():
    g = dict(zip(REAL_TRAIN_IDS, site_groups(REAL_TRAIN_IDS)))
    assert len({g[i] for i in ["H366", "H367", "H368", "H371", "H372", "H374"]}) == 1
    assert g["H181"] == g["H183"] and g["H615"] == g["H617"] and g["H666"] == g["H668"]
    assert g["H126"] != g["H181"] and g["F827"] != g["G190"]
    assert len(set(g.values())) == 13


def test_grouped_kfold_never_splits_groups():
    groups = site_groups(REAL_TRAIN_IDS)
    for trn, val in grouped_kfold(groups, 5):
        assert not set(groups[trn]) & set(groups[val])


def test_make_valid_and_submission(tmp_path):
    raw = np.array([[5, 3, 10, 120, 50, 60, 70, 80, 90, 95, 99.99999], [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], float)
    v = make_valid(raw)
    assert np.all(np.diff(v, axis=1) >= 0) and np.all(v[:, -1] == 100.0) and v.min() >= 0 and v.max() <= 100
    df = write_submission(["a", "b"], raw, tmp_path / "s.csv")
    back = pd.read_csv(tmp_path / "s.csv")
    assert list(back.columns)[1:] == ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
    validate_submission(back, ["a", "b"])
    bad = df.copy()
    bad.loc[0, "200"] = 99.9
    with pytest.raises(SubmissionError):
        validate_submission(bad, ["a", "b"])
    with pytest.raises(SubmissionError):
        validate_submission(build_submission(["a", "b"], raw), ["a", "c"])


def test_pointwise_median_monotone():
    rng = np.random.default_rng(1)
    curves = [make_valid(np.sort(rng.uniform(0, 100, (4, 11)), axis=1)) for _ in range(5)]
    med = pointwise_median(curves)
    assert np.all(np.diff(med, axis=1) >= -1e-12)
    wmed = pointwise_median(curves, [1, 1, 5, 1, 1])
    assert np.allclose(wmed, curves[2])


def test_clr_roundtrip():
    y = np.array(list(REAL_LABELS.values()), float)
    back = clr_to_cdf(cdf_to_clr(y))
    assert np.abs(back - y).max() < 1e-6


def test_classical_models_fit_predict():
    rng = np.random.default_rng(0)
    y = np.array(list(REAL_LABELS.values()), float)
    X = rng.normal(size=(len(y), 6))
    for m in (CLRRegressor("ridge"), CLRRegressor("pls", n_components=2), KNNMedian(3)):
        p = m.fit(X, y).predict(X)
        assert p.shape == y.shape and np.all(np.diff(p, axis=1) >= 0) and np.all(p[:, -1] == 100)


def test_features_are_finite(fake_root):
    from phygrainnet.data.imaging import canonical_image

    cat = build_catalog(fake_root)
    r = cat.iloc[0]
    img = canonical_image(r.path, r.ppm, 2.0, 0.0)
    f = image_features(img, 2.0)
    assert len(f) > 60 and all(np.isfinite(v) for v in f.values())


def test_config_inheritance(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "phygrainnet" / "mv_ets.yaml",
                      ["training.epochs=3", "data.fields_mm=[10,20]"])
    assert cfg["training"]["epochs"] == 3 and cfg["data"]["fields_mm"] == [10, 20]
    assert cfg["data"]["target_ppm"] == 10.0 and cfg["model"]["pretrained"] is False
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}}


def test_mv_model_valid_output_and_variable_tiles():
    m = PhyGrainNetMV(num_scales=2, channels=(8, 16), blocks=(1, 1), embed_dim=16).eval()
    for t in (1, 5):
        out = m(torch.randn(2, t, 2, 3, 32, 32))["cumulative"]
        assert out.shape == (2, 11)
        assert torch.all(out[:, 1:] >= out[:, :-1] - 1e-5)
        assert torch.allclose(out[:, -1], torch.full((2,), 100.0), atol=1e-3)


def test_classical_pipeline_end_to_end(tiny_cfg):
    from phygrainnet.training.pipelines import run_classical

    cfg = copy.deepcopy(tiny_cfg)
    cfg["cv"]["strategy"] = "logo"
    cfg["experiment_id"] = "A001"
    cfg["classical"] = {"models": [{"name": "ridge", "kind": "ridge_clr", "alpha": 10}, {"name": "prior", "kind": "prior_median"}]}
    res = run_classical(cfg)
    assert set(res) == {"A001_ridge", "A001_prior"}
    out = Path(cfg["output"]["dir"])
    test = pd.read_csv(out / "A001_ridge" / "test.csv")
    assert test["sample_id"].tolist() == TEST_IDS
    assert np.isfinite(res["A001_ridge"]["cv_emd"])
    assert pd.read_csv(cfg["output"]["registry"]).shape[0] == 2


def test_cnn_pipeline_ssl_and_init(tiny_cfg):
    from phygrainnet.training.pipelines import run_cnn, run_ssl
    from phygrainnet.training.trainer import load_checkpoint

    cfg = copy.deepcopy(tiny_cfg)
    cfg["experiment_id"] = "P000"
    cfg["ssl"] = {"epochs": 1, "steps_per_epoch": 2, "batch_size": 4, "tile_px": 32, "use_competition_photos": True,
                  "include_test_photos": True, "use_ets": False}
    ckpt = run_ssl(cfg)
    assert load_checkpoint(ckpt)["origin"] == "phygrainnet-from-scratch"

    cfg2 = copy.deepcopy(tiny_cfg)
    cfg2["experiment_id"] = "D100"
    cfg2["training"].update(init_checkpoint=ckpt, init_encoder_only=True, full_fit=True)
    cfg2["cv"]["max_folds"] = 1
    summary = run_cnn(cfg2)
    assert np.isfinite(summary["cv_emd"])
    test = pd.read_csv(Path(cfg2["output"]["dir"]) / "D100" / "test.csv")
    v = test.iloc[:, 1:].to_numpy()
    assert np.all(np.diff(v, axis=1) >= 0) and np.all(v[:, -1] == 100)


def test_foreign_checkpoint_rejected(tmp_path):
    from phygrainnet.training.trainer import load_checkpoint

    torch.save({"state_dict": {}}, tmp_path / "imagenet.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "imagenet.pt")


def test_camera_name_variants(fake_root):
    from phygrainnet.data.catalog import CameraTable, parse_photo_name

    cams = CameraTable.from_csv(fake_root / "ppm_updated.csv")
    ids = list(REAL_LABELS)
    assert parse_photo_name("Samsung_Galaxy_A52_H405_01", cams, ids)[:2] == ("Samsung A52", "H405")
    assert parse_photo_name("SM-A525F_H405_02", cams, ids)[:2] == ("Samsung A52", "H405")
    assert parse_photo_name("H405_SamsungA52_3", cams, ids)[:2] == ("Samsung A52", "H405")
    assert parse_photo_name("Motorola_Edge_60_fusion_H374_01", cams, ids)[:2] == ("Motorola Edge 60 Fusion", "H374")
    assert parse_photo_name("iPhone16_HPC_Kleinkummerfeld 2-3 (1)", cams, TEST_IDS)[1] == "HPC_Kleinkummerfeld 2-3"
