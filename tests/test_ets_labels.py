import numpy as np
import pandas as pd
import torch

from phygrainnet.data.ets_labels import (
    TARGET_COLUMNS,
    convert_dataframe,
    interpolate_curve,
    parse_sieve_header,
)
from phygrainnet.losses import masked_emd_loss
from phygrainnet.metrics.kaggle_emd import provisional_log_grid_emd

BNQ_MM = [80, 56, 40, 31.5, 20, 14, 10, 5, 2.5, 1.25, 0.63, 0.315, 0.16, 0.08]


def test_header_parsing():
    assert parse_sieve_header("80 mm") == 80
    assert parse_sieve_header("80 µm") == 0.08
    assert parse_sieve_header("0,630") == 0.63
    assert parse_sieve_header("Tamis 2.5 mm") == 2.5
    assert parse_sieve_header("Sample 12") is None
    assert parse_sieve_header(5) == 5.0


def test_interpolation_hits_exact_sieve_and_masks_fines():
    sieves = np.array(BNQ_MM, dtype=float)
    passing = np.linspace(100, 5, len(sieves))  # coarse -> fine
    curve = interpolate_curve(sieves, passing)
    # 20 mm and 0.63 mm are real sieves -> exact
    assert np.isclose(curve[TARGET_COLUMNS.index("20")], passing[BNQ_MM.index(20)])
    assert np.isclose(curve[TARGET_COLUMNS.index("0.63")], passing[BNQ_MM.index(0.63)])
    # below 0.08 mm unsupported
    assert np.all(np.isnan(curve[:4]))
    assert curve[-1] == 100.0


def test_convert_dataframe_fraction_and_monotone_fix():
    cols = [f"{s} mm" for s in BNQ_MM]
    row = np.linspace(1.0, 0.05, len(BNQ_MM))
    row[5] = row[4] + 0.01  # small non-monotone glitch
    df = pd.DataFrame([["S1", *row]], columns=["Sample", *cols])
    out, rep = convert_dataframe(df)
    assert rep.percent_was_fraction
    assert rep.rows_with_monotone_fixes == 1
    vals = out[TARGET_COLUMNS].to_numpy(dtype=float)[0]
    finite = vals[np.isfinite(vals)]
    assert np.all(np.diff(finite) >= -1e-9)
    assert out["mask_0.063"].iloc[0] == 0 and out["mask_0.2"].iloc[0] == 1


def test_masked_loss_equals_official_metric_when_fully_labelled():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 100, (5, 11)), axis=1); a[:, -1] = 100
    b = np.sort(rng.uniform(0, 100, (5, 11)), axis=1); b[:, -1] = 100
    ref = provisional_log_grid_emd(a, b).mean()
    got = masked_emd_loss(torch.tensor(b, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)).item()
    assert np.isclose(ref, got, rtol=1e-5)


def test_upper_bound_penalises_fines_above_p80():
    target = torch.tensor([[np.nan] * 4 + [30, 50, 70, 85, 95, 100, 100]], dtype=torch.float32)
    mask = torch.tensor([[0] * 4 + [1] * 7])
    pred_ok = torch.tensor([[1, 2, 5, 10, 30, 50, 70, 85, 95, 100, 100]], dtype=torch.float32)
    pred_bad = pred_ok.clone(); pred_bad[0, 3] = 25.0  # above P80 = 12
    kw = dict(mask=mask, upper_bound=torch.tensor([12.0]), finest_index=torch.tensor([4]))
    assert masked_emd_loss(pred_ok, target, **kw).item() == 0.0
    assert masked_emd_loss(pred_bad, target, **kw).item() > 0.0
