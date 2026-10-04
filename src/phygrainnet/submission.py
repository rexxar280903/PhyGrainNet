"""Write and validate Kaggle submission files exactly as the scorer expects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phygrainnet.constants import TARGET_COLUMNS
from phygrainnet.postprocess import make_valid


class SubmissionError(ValueError):
    pass


def validate_submission(df: pd.DataFrame, test_ids: list[str]) -> None:
    """Mirror the rules on the Data/Evaluation pages; raise on the first violation."""
    expected_cols = ["sample_id", *TARGET_COLUMNS]
    if list(df.columns) != expected_cols:
        raise SubmissionError(f"columns must be {expected_cols}, got {list(df.columns)}")
    ids = df["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise SubmissionError("duplicate sample_id rows")
    if set(ids) != set(test_ids):
        raise SubmissionError(
            f"id mismatch: missing {sorted(set(test_ids) - set(ids))}, extra {sorted(set(ids) - set(test_ids))}"
        )
    v = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise SubmissionError("non-finite values")
    if v.min() < 0 or v.max() > 100:
        raise SubmissionError("values outside [0, 100]")
    if np.any(np.diff(v, axis=1) < 0):
        bad = df.loc[np.any(np.diff(v, axis=1) < 0, axis=1), "sample_id"].tolist()
        raise SubmissionError(f"non-monotone rows: {bad}")
    if not np.all(v[:, -1] == 100.0):
        raise SubmissionError("200 mm column must be exactly 100")


def build_submission(test_ids: list[str], preds: np.ndarray) -> pd.DataFrame:
    preds = make_valid(preds)
    if preds.shape != (len(test_ids), len(TARGET_COLUMNS)):
        raise SubmissionError(f"prediction shape {preds.shape} does not match {len(test_ids)} ids")
    df = pd.DataFrame(preds, columns=TARGET_COLUMNS)
    df.insert(0, "sample_id", test_ids)
    return df


def write_submission(test_ids: list[str], preds: np.ndarray, path: str | Path) -> pd.DataFrame:
    df = build_submission(test_ids, preds)
    validate_submission(df, test_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    # Re-read to make sure what is on disk also validates (float formatting etc.).
    back = pd.read_csv(path)
    back.columns = ["sample_id", *TARGET_COLUMNS]
    validate_submission(back, test_ids)
    return df


def read_prediction_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["sample_id", *[f"{float(c):g}" for c in df.columns[1:]]]
    df["sample_id"] = df["sample_id"].astype(str)
    return df
