"""Competition constants verified against the live Kaggle pages (2026-10-04)."""
from __future__ import annotations

import numpy as np

# DIN EN ISO 14688-1 support points, mm.
DIAMETERS_MM = np.array(
    [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0],
    dtype=np.float64,
)

# Header names exactly as in sample_submission.csv / Training_labels_updated.csv.
TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]

# Official per-interval weights: log10(x_{i+1}) - log10(x_i) (10 intervals, ~0.5 each).
EMD_WEIGHTS = np.diff(np.log10(DIAMETERS_MM))

NUM_POINTS = len(DIAMETERS_MM)

# Kaggle mount and real file names.
DEFAULT_DATA_ROOT = "/kaggle/input/soil-grain-size-from-photos"
TRAIN_LABELS_FILE = "Training_labels_updated.csv"
PPM_FILE = "ppm_updated.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"
TRAIN_PHOTO_DIR = "Training-All_Photos_updated"
TEST_PHOTO_DIR = "Test_All_Photos"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

COMPETITION_DEADLINE = "2026-11-30T11:00:00Z"  # 18:00 WIB
