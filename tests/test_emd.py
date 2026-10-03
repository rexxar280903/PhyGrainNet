import numpy as np

from phygrainnet.metrics.kaggle_emd import mean_provisional_log_grid_emd


def test_identical_curve_has_zero_provisional_emd():
    curve = np.array([[0, 1, 3, 8, 15, 30, 55, 75, 90, 97, 100]], dtype=float)
    assert mean_provisional_log_grid_emd(curve, curve) == 0.0


def test_different_curve_has_positive_provisional_emd():
    a = np.array([[0, 1, 3, 8, 15, 30, 55, 75, 90, 97, 100]], dtype=float)
    b = np.array([[0, 2, 5, 12, 22, 40, 65, 82, 94, 99, 100]], dtype=float)
    assert mean_provisional_log_grid_emd(a, b) > 0.0
