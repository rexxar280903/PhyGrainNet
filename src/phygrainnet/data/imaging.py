"""Image loading, physical-scale normalisation, colour normalisation, augmentation.

Everything works on uint8 RGB numpy arrays. Paths may contain non-ASCII characters
(`Münster`), so decoding goes through PIL / np.fromfile instead of cv2.imread.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

cv2.setNumThreads(1)


def load_rgb(path: str | Path, exif_transpose: bool = True) -> np.ndarray:
    with Image.open(path) as im:
        if exif_transpose:
            im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"))


def crop_border(img: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0:
        return img
    h, w = img.shape[:2]
    dy, dx = int(round(h * frac)), int(round(w * frac))
    return img[dy : h - dy, dx : w - dx]


def to_ppm(img: np.ndarray, ppm: float, target_ppm: float) -> np.ndarray:
    """Resample so that 1 mm = target_ppm pixels."""
    scale = target_ppm / ppm
    if abs(scale - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, size, interpolation=interp)


def normalize_color(img: np.ndarray, mode: str = "grayworld") -> np.ndarray:
    """Reduce camera colour differences (train: Motorola/Samsung, test: iPhone).

    grayworld : per-channel gain so channel means are equal (white balance only).
    lab       : standardise L*a*b* mean/std to fixed reference values.
    none      : unchanged.
    """
    if mode == "none":
        return img
    x = img.astype(np.float32)
    if mode == "grayworld":
        means = x.reshape(-1, 3).mean(axis=0) + 1e-6
        x = x * (means.mean() / means)
        return np.clip(x, 0, 255).astype(np.uint8)
    if mode == "lab":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_mean = np.array([128.0, 128.0, 135.0], dtype=np.float32)
        ref_std = np.array([40.0, 4.0, 8.0], dtype=np.float32)
        m = lab.reshape(-1, 3).mean(0)
        s = lab.reshape(-1, 3).std(0) + 1e-6
        # keep chroma differences (they carry soil information): only shift a/b, scale L
        lab[..., 0] = (lab[..., 0] - m[0]) / s[0] * ref_std[0] + ref_mean[0]
        lab[..., 1:] = lab[..., 1:] - m[1:] + ref_mean[1:]
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    raise ValueError(f"unknown colour mode {mode!r}")


def canonical_image(
    path: str | Path,
    ppm: float,
    target_ppm: float,
    border_frac: float = 0.06,
    color_mode: str = "grayworld",
) -> np.ndarray:
    img = load_rgb(path)
    img = crop_border(img, border_frac)
    img = to_ppm(img, ppm, target_ppm)
    return normalize_color(img, color_mode)


def cache_path(cache_dir: str | Path, path: str | Path, target_ppm: float, border_frac: float, color_mode: str) -> Path:
    key = f"{Path(path).name}|{target_ppm}|{border_frac}|{color_mode}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{h}.png"


def canonical_image_cached(
    path: str | Path,
    ppm: float,
    target_ppm: float,
    border_frac: float = 0.06,
    color_mode: str = "grayworld",
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    if cache_dir is None:
        return canonical_image(path, ppm, target_ppm, border_frac, color_mode)
    cp = cache_path(cache_dir, path, target_ppm, border_frac, color_mode)
    if cp.exists():
        return np.asarray(Image.open(cp).convert("RGB"))
    img = canonical_image(path, ppm, target_ppm, border_frac, color_mode)
    cp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(cp, compress_level=1)
    return img


# ----------------------------------------------------------------------------- augmentation

def random_crop(img: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        h, w = img.shape[:2]
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return img[y : y + size, x : x + size]


def crop_at(img: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2
    y0 = int(np.clip(cy - half, 0, max(0, h - size)))
    x0 = int(np.clip(cx - half, 0, max(0, w - size)))
    out = img[y0 : y0 + size, x0 : x0 + size]
    if out.shape[0] != size or out.shape[1] != size:
        out = cv2.copyMakeBorder(out, 0, size - out.shape[0], 0, size - out.shape[1], cv2.BORDER_REFLECT_101)
    return out


def multiscale_tile(
    img: np.ndarray,
    cy: int,
    cx: int,
    fields_px: list[int],
    out_px: int,
) -> list[np.ndarray]:
    """Concentric crops of different physical fields, each resized to out_px."""
    tiles = []
    for f in fields_px:
        c = crop_at(img, cy, cx, f)
        if f != out_px:
            c = cv2.resize(c, (out_px, out_px), interpolation=cv2.INTER_AREA if f > out_px else cv2.INTER_LINEAR)
        tiles.append(c)
    return tiles


def geometric_aug(tile: np.ndarray, k: int, flip: bool) -> np.ndarray:
    t = np.rot90(tile, k)
    if flip:
        t = t[:, ::-1]
    return np.ascontiguousarray(t)


def photometric_aug(tile: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Simulate phone differences: white balance, exposure, contrast, saturation, blur, JPEG, noise."""
    x = tile.astype(np.float32)
    gains = 1.0 + strength * rng.uniform(-0.08, 0.08, size=3)
    x = x * gains
    x = (x - 128.0) * (1.0 + strength * rng.uniform(-0.15, 0.15)) + 128.0 + strength * rng.uniform(-15, 15)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * (1.0 + strength * rng.uniform(-0.2, 0.2))
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.3 * strength:
        sigma = rng.uniform(0.3, 1.0)
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    if rng.random() < 0.4 * strength:
        q = int(rng.integers(55, 96))
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    if rng.random() < 0.3 * strength:
        x = np.clip(x.astype(np.float32) + rng.normal(0, rng.uniform(1, 5), x.shape), 0, 255).astype(np.uint8)
    return x


def grid_centers(h: int, w: int, step: int, margin: int) -> list[tuple[int, int]]:
    ys = list(range(margin, max(margin + 1, h - margin), step)) or [h // 2]
    xs = list(range(margin, max(margin + 1, w - margin), step)) or [w // 2]
    return [(y, x) for y in ys for x in xs]
