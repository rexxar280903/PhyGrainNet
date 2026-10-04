"""Synthetic stand-in for the Kaggle dataset, using the real file-name conventions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phygrainnet.constants import TARGET_COLUMNS  # noqa: E402

REAL_LABELS = {
    "F827": [9.4904, 19.3892, 47.6257, 89.5554, 99.8965, 99.9896, 100, 100, 100, 100, 100],
    "H366": [0.9633, 2.0893, 5.0347, 9.4833, 13.7489, 20.2655, 32.817, 55.3867, 86.2023, 100, 100],
    "H367": [0.8232, 1.8584, 4.1483, 7.5417, 11.0995, 17.3609, 30.4114, 56.1341, 85.7588, 100, 100],
    "H374": [1.1223, 2.4374, 5.5348, 9.3031, 13.1208, 18.2481, 29.4152, 51.2113, 76.4875, 92.6403, 100],
    "H405": [17.7068, 26.4625, 47.111, 75.656, 91.3413, 95.9413, 98.4538, 100, 100, 100, 100],
    "H493": [3.275, 4.8317, 15.1817, 51.9481, 95.7849, 98.8004, 99.2105, 99.7965, 100, 100, 100],
    "H616": [0, 0, 0, 2.7806, 47.7306, 97.5687, 98.3221, 98.6542, 99.3582, 100, 100],
    "H637": [2.0416, 3.623, 6.3369, 14.8196, 24.9023, 35.4016, 46.8916, 61.9279, 81.061, 100, 100],
}
TEST_IDS = ["HPC_Muenster_BS6_9_0-10m", "HPC_Kleinkummerfeld 2-2", "HPC_Kleinkummerfeld 2-3"]

CAMERAS = pd.DataFrame(
    {
        "phone": ["iPhone 14", "iPhone 16", "Motorola Edge", "Motorola Edge 60 Fusion", "Samsung A52"],
        "camera": ["iPhone 14", "iPhone 16", "motorola edge 20", "Motorola Edge 60 fusion", "SM-A525F"],
        "width": [240, 240, 240, 180, 240],
        "height": [180, 180, 120, 240, 180],
        "ppm": [2.0, 2.2, 1.8, 2.0, 2.4],
    }
)


def _texture(rng: np.random.Generator, w: int, h: int, grain_px: float) -> np.ndarray:
    img = np.full((h, w, 3), 120, np.float32) + rng.normal(0, 8, (h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(int(w * h / max(4.0, grain_px**2) * 0.6)):
        r = max(1.0, rng.gamma(2.0, grain_px / 2))
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        img[m] = rng.uniform(60, 220, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("kaggle_input") / "soil-grain-size-from-photos"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    CAMERAS.to_csv(root / "ppm_updated.csv", index=False)
    lab = pd.DataFrame([[k, *v] for k, v in REAL_LABELS.items()], columns=["sample_id", *TARGET_COLUMNS])
    lab.to_csv(root / "Training_labels_updated.csv", index=False)
    sub = pd.DataFrame([[t, *([0.0] * 11)] for t in TEST_IDS], columns=["sample_id", *TARGET_COLUMNS])
    sub.to_csv(root / "sample_submission.csv", index=False)

    tr = root / "Training-All_Photos_updated" / "Training-All_Photos_updated"
    te = root / "Test_All_Photos" / "Test_All_Photos"
    tr.mkdir(parents=True)
    te.mkdir(parents=True)
    cam_for = {"H374": ("Motorola_Edge_60_fusion", 180, 240), "H405": ("Samsung_A52", 240, 180)}
    for sid, curve in REAL_LABELS.items():
        prefix, w, h = cam_for.get(sid, ("Motorola_Edge", 240, 120))
        d50 = float(np.interp(50, curve, np.log10([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])))
        grain = 1.5 + 4 * (d50 + 3) / 5
        for v in (1, 2):
            Image.fromarray(_texture(rng, w, h, grain)).save(tr / f"{prefix}_{sid}_{v:02d}.jpg", quality=90)
    names = {
        "HPC_Muenster_BS6_9_0-10m": ["iPhone14_HPC_Münster_BS6_9,0-10m (1).JPG", "iPhone14_HPC_Münster_BS6_9,0-10m (2).JPG"],
        "HPC_Kleinkummerfeld 2-2": ["iPhone16_HPC_Kleinkummerfeld 2-2 (2).JPG", "iPhone16_HPC_Kleinkummerfeld 2-2 (3).JPG"],
        "HPC_Kleinkummerfeld 2-3": ["iPhone16_HPC_Kleinkummerfeld 2-3 (1).JPG"],
    }
    for sid, files in names.items():
        for f in files:
            Image.fromarray(_texture(rng, 240, 180, 3.0)).save(te / f, quality=90)
    return root


@pytest.fixture()
def tiny_cfg(fake_root, tmp_path) -> dict:
    from phygrainnet.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    cfg["data"].update(
        root=str(fake_root), cache_dir=str(tmp_path / "cache"), target_ppm=2.0, fields_mm=[16.0, 32.0], tile_px=32,
        num_workers=0, n_jobs=1, border_frac=0.0,
    )
    cfg["model"].update(channels=[8, 16, 24], blocks=[1, 1, 1], embed_dim=32)
    cfg["training"].update(epochs=1, steps_per_epoch=2, batch_size=2, tiles_per_sample=3, warmup_steps=1, eval_every=1)
    cfg["inference"].update(tta=2, max_tiles_per_image=4, chunk=8)
    cfg["output"].update(dir=str(tmp_path / "outputs"), registry=str(tmp_path / "registry.csv"), save_checkpoints=True)
    cfg["cv"].update(strategy="group_kfold", n_folds=2)
    return cfg
