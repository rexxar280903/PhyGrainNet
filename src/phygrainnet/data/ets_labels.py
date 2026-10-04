"""Convert external PSD labels (e.g. the ETS photogranulometry dataset) to the
11 competition support points.

The competition host interpolated raw laboratory curves *linearly on a
log10(d) axis* onto the 11 DIN EN ISO 14688-1 diameters. We do exactly the
same here so that external labels live on the same grid as the competition
labels.

Points that the external data cannot support (below its finest sieve) are
returned as NaN together with a 0 in the matching mask column, so the training
loss can ignore them. The % passing the finest sieve is kept as an upper bound
for all finer points (a cumulative curve can only decrease towards finer d).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_DIAMETERS_MM = np.array(
    [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0],
    dtype=np.float64,
)
# Column names exactly as in the competition's sample_submission.csv.
TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]

_SIZE_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|µm|μm|um|micron|microns)?",
    re.IGNORECASE,
)
_ID_HINTS = ("sample", "échantillon", "echantillon", "id", "no", "numéro", "numero", "name")


@dataclass
class ConversionReport:
    n_rows_in: int
    n_rows_out: int
    sieves_mm: list[float]
    id_column: str
    percent_was_fraction: bool
    rows_with_monotone_fixes: int
    rows_dropped: list[str]
    supported_points: list[str]

    def __str__(self) -> str:
        lines = [
            f"rows in / out           : {self.n_rows_in} / {self.n_rows_out}",
            f"id column               : {self.id_column!r}",
            f"sieves detected (mm)    : {self.sieves_mm}",
            f"values given as 0-1     : {self.percent_was_fraction}",
            f"rows needing monotone fix: {self.rows_with_monotone_fixes}",
            f"supported target points : {self.supported_points}",
        ]
        if self.rows_dropped:
            lines.append(f"dropped rows            : {self.rows_dropped[:20]}{' ...' if len(self.rows_dropped) > 20 else ''}")
        return "\n".join(lines)


def parse_sieve_header(header: object, default_unit: str = "mm") -> float | None:
    """Return the sieve opening in mm encoded in a column header, or None.

    Accepts e.g. '80 mm', '0,080', '80 µm', '5', 'Tamis 2.5 mm'. A bare number
    is interpreted with ``default_unit``.
    """
    if isinstance(header, (int, float, np.integer, np.floating)) and not isinstance(header, bool):
        value = float(header)
        return value / 1000.0 if default_unit.lower() in {"um", "µm", "μm"} else value
    text = str(header).strip()
    match = _SIZE_RE.search(text)
    if not match:
        return None
    # Reject headers where the number is clearly not a sieve size (e.g. "Sample 12").
    leftover = text[: match.start()].strip().lower()
    if any(hint in leftover for hint in ("sample", "échantillon", "echantillon", "id", "no")):
        return None
    value = float(match.group("num").replace(",", "."))
    unit = (match.group("unit") or default_unit).lower()
    if unit in {"µm", "μm", "um", "micron", "microns"}:
        value /= 1000.0
    return value if value > 0 else None


def _find_id_column(df: pd.DataFrame, sieve_cols: list) -> str:
    candidates = [c for c in df.columns if c not in sieve_cols]
    for c in candidates:
        if any(h in str(c).lower() for h in _ID_HINTS):
            return c
    if candidates:
        return candidates[0]
    raise ValueError("could not find a sample-id column")


def interpolate_curve(
    sieves_mm: np.ndarray,
    passing: np.ndarray,
    targets_mm: np.ndarray = TARGET_DIAMETERS_MM,
) -> np.ndarray:
    """Log10-linear interpolation of one cumulative curve.

    * inside the measured range -> interpolated
    * above the coarsest sieve -> 100 if the curve already reached 100, else NaN
      (except the 200 mm point, which is 100 by competition definition)
    * below the finest sieve -> NaN (unsupported)
    """
    order = np.argsort(sieves_mm)
    x = np.log10(np.asarray(sieves_mm, dtype=np.float64)[order])
    y = np.asarray(passing, dtype=np.float64)[order]
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    out = np.full(targets_mm.shape, np.nan)
    if x.size < 2:
        return out
    lt = np.log10(targets_mm)
    inside = (lt >= x[0]) & (lt <= x[-1])
    out[inside] = np.interp(lt[inside], x, y)
    above = lt > x[-1]
    if y[-1] >= 100.0 - 1e-6:
        out[above] = 100.0
    out[np.isclose(targets_mm, 200.0)] = 100.0
    return out


def convert_dataframe(
    df: pd.DataFrame,
    id_column: str | None = None,
    default_unit: str = "mm",
    min_sieves: int = 4,
) -> tuple[pd.DataFrame, ConversionReport]:
    sieve_map = {c: parse_sieve_header(c, default_unit) for c in df.columns}
    sieve_cols = [c for c, v in sieve_map.items() if v is not None]
    if id_column is None:
        id_column = _find_id_column(df, sieve_cols)
    sieve_cols = [c for c in sieve_cols if c != id_column]
    if len(sieve_cols) < min_sieves:
        raise ValueError(
            f"only {len(sieve_cols)} sieve columns detected: {sieve_cols}. "
            "Pass the sheet/header row explicitly or rename columns to e.g. '80 mm'."
        )
    sieves = np.array([sieve_map[c] for c in sieve_cols], dtype=np.float64)
    order = np.argsort(sieves)
    sieve_cols = [sieve_cols[i] for i in order]
    sieves = sieves[order]

    values = df[sieve_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    as_fraction = finite.size > 0 and np.nanmax(finite) <= 1.0 + 1e-9
    if as_fraction:
        values = values * 100.0
    values = np.clip(values, 0.0, 100.0)

    rows, dropped, fixes = [], [], 0
    for sid, row in zip(df[id_column].astype(str), values):
        if np.isfinite(row).sum() < min_sieves:
            dropped.append(sid)
            continue
        # Enforce monotonicity over sieves that were measured (cumulative max).
        filled = row.copy()
        ok = np.isfinite(filled)
        mono = np.maximum.accumulate(filled[ok])
        if not np.allclose(mono, filled[ok]):
            fixes += 1
        filled[ok] = mono
        curve = interpolate_curve(sieves, filled)
        finest = sieves[ok][0]
        p_finest = filled[ok][0]
        rec = {"sample_id": sid}
        rec.update({col: curve[i] for i, col in enumerate(TARGET_COLUMNS)})
        rec.update({f"mask_{col}": int(np.isfinite(curve[i])) for i, col in enumerate(TARGET_COLUMNS)})
        rec["finest_sieve_mm"] = finest
        rec["passing_finest_sieve"] = p_finest  # upper bound for all finer points
        rec["n_sieves"] = int(ok.sum())
        rows.append(rec)

    out = pd.DataFrame(rows)
    supported = [c for c in TARGET_COLUMNS if out.get(f"mask_{c}", pd.Series(dtype=int)).mean() > 0.5] if len(out) else []
    report = ConversionReport(
        n_rows_in=len(df),
        n_rows_out=len(out),
        sieves_mm=[float(s) for s in sieves],
        id_column=str(id_column),
        percent_was_fraction=bool(as_fraction),
        rows_with_monotone_fixes=fixes,
        rows_dropped=dropped,
        supported_points=supported,
    )
    return out, report


def load_table(path: Path, sheet: str | int | None = 0, header_row: int = 0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet, header=header_row)
    return pd.read_csv(path, header=header_row, sep=None, engine="python")


def roundtrip_error(
    labels_11pt: np.ndarray,
    sieves_mm: np.ndarray,
) -> dict[str, float]:
    """Estimate interpolation noise: resample competition curves to an external
    sieve set and back, then measure the official EMD on supported points only.
    """
    w = np.diff(np.log10(TARGET_DIAMETERS_MM))
    errs, per_point = [], []
    for y in labels_11pt:
        at_sieves = np.interp(np.log10(sieves_mm), np.log10(TARGET_DIAMETERS_MM), y)
        back = interpolate_curve(sieves_mm, at_sieves)
        diff = np.abs(back[:-1] - y[:-1])
        diff = np.where(np.isfinite(diff), diff, 0.0)
        per_point.append(diff * w)
        errs.append(float(np.sum(diff * w)))
    per_point = np.mean(per_point, axis=0)
    return {"mean_emd_on_supported_points": float(np.mean(errs)),
            **{f"pt_{c}": float(v) for c, v in zip(TARGET_COLUMNS[:-1], per_point)}}
