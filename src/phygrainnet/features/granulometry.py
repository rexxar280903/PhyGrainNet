"""Hand-crafted, physically scaled image descriptors (no learned weights).

All radii / wavelengths are given in millimetres and converted with the image's
canonical PPM, so the same feature means the same physical scale on every phone.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

DEFAULT_RADII_MM = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6)
DEFAULT_BANDS_MM = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _luminance(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)


def _disk(radius_px: int) -> np.ndarray:
    d = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _morph_at_scale(lum: np.ndarray, radius_px: float, op: int, max_kernel_radius: int = 6) -> float:
    """Mean of opening/closing with a disk of `radius_px`, using a pyramid so the
    kernel never exceeds `max_kernel_radius` pixels (keeps 25 mm radii cheap)."""
    if radius_px < 0.5:
        return float(lum.mean())
    factor = max(1.0, radius_px / max_kernel_radius)
    if factor > 1.0:
        small = cv2.resize(lum, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
        r = max(1, int(round(radius_px / factor)))
    else:
        small, r = lum, max(1, int(round(radius_px)))
    out = cv2.morphologyEx(small, op, _disk(r))
    return float(out.mean())


def pattern_spectra(lum: np.ndarray, ppm: float, radii_mm=DEFAULT_RADII_MM) -> dict[str, float]:
    """Granulometric pattern spectrum: how much bright (opening) and dark (closing)
    structure disappears between consecutive disk sizes. Bright grains of radius r
    vanish under an opening with a disk larger than r."""
    base = float(lum.mean()) + 1e-6
    feats: dict[str, float] = {}
    prev_o, prev_c = base, base
    for r in radii_mm:
        o = _morph_at_scale(lum, r * ppm, cv2.MORPH_OPEN)
        c = _morph_at_scale(lum, r * ppm, cv2.MORPH_CLOSE)
        feats[f"open_{r:g}mm"] = (prev_o - o) / base
        feats[f"close_{r:g}mm"] = (c - prev_c) / base
        prev_o, prev_c = o, c
    feats["open_residual"] = prev_o / base
    feats["close_residual"] = prev_c / base
    return feats


def radial_spectrum(lum: np.ndarray, ppm: float, bands_mm=DEFAULT_BANDS_MM, max_side: int = 2048) -> dict[str, float]:
    """Share of luminance variance per wavelength band (in mm)."""
    h, w = lum.shape
    s = min(h, w, max_side)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = lum[y0 : y0 + s, x0 : x0 + s]
    crop = crop - crop.mean()
    win = np.outer(np.hanning(s), np.hanning(s)).astype(np.float32)
    f = np.fft.rfft2(crop * win)
    p = (np.abs(f) ** 2).astype(np.float64)
    fy = np.fft.fftfreq(s)[:, None]
    fx = np.fft.rfftfreq(s)[None, :]
    freq = np.sqrt(fx**2 + fy**2)  # cycles per pixel
    with np.errstate(divide="ignore"):
        wavelength_mm = np.where(freq > 0, 1.0 / (freq * ppm), np.inf)
    total = p[np.isfinite(wavelength_mm)].sum() + 1e-12
    edges = list(bands_mm) + [np.inf]
    feats = {}
    nyquist_mm = 2.0 / ppm
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (wavelength_mm >= max(lo, nyquist_mm)) & (wavelength_mm < hi)
        feats[f"fft_{lo:g}mm"] = float(p[m].sum() / total) if m.any() else 0.0
    return feats


def gradient_scale_space(lum: np.ndarray, ppm: float, sigmas_mm=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)) -> dict[str, float]:
    feats = {}
    for s in sigmas_mm:
        sp = s * ppm
        factor = max(1.0, sp / 3.0)
        small = cv2.resize(lum, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA) if factor > 1 else lum
        sp_small = max(0.5, sp / factor)
        g = cv2.GaussianBlur(small, (0, 0), sp_small)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2) * sp_small  # scale-normalised
        feats[f"grad_{s:g}mm"] = float(mag.mean())
        lap = cv2.Laplacian(g, cv2.CV_32F) * sp_small**2
        feats[f"blob_{s:g}mm"] = float(np.abs(lap).mean())
    return feats


def color_stats(img: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32).reshape(-1, 3)
    feats = {}
    for i, n in enumerate("Lab"):
        feats[f"{n}_mean"] = float(lab[:, i].mean())
        feats[f"{n}_std"] = float(lab[:, i].std())
    for q in (5, 25, 50, 75, 95):
        feats[f"L_p{q}"] = float(np.percentile(lab[:, 0], q))
    feats["sat_mean"] = float(hsv[:, 1].mean())
    feats["dark_frac"] = float((lab[:, 0] < 50).mean())
    feats["bright_frac"] = float((lab[:, 0] > 200).mean())
    return feats


def lbp_hist(lum: np.ndarray, ppm: float, radii_mm=(0.2, 1.0, 4.0)) -> dict[str, float]:
    feats = {}
    for r in radii_mm:
        factor = max(1.0, r * ppm / 2.0)
        small = cv2.resize(lum, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA) if factor > 1 else lum
        codes = local_binary_pattern(small.astype(np.uint8), P=8, R=2, method="uniform")
        hist = np.bincount(codes.astype(int).ravel(), minlength=10)[:10].astype(np.float64)
        hist /= hist.sum() + 1e-12
        for k, v in enumerate(hist):
            feats[f"lbp{r:g}_{k}"] = float(v)
    return feats


def image_features(img: np.ndarray, ppm: float) -> dict[str, float]:
    lum = _luminance(img)
    feats: dict[str, float] = {}
    feats.update(color_stats(img))
    feats.update(pattern_spectra(lum, ppm))
    feats.update(radial_spectrum(lum, ppm))
    feats.update(gradient_scale_space(lum, ppm))
    feats.update(lbp_hist(lum, ppm))
    return feats
