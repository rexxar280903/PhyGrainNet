"""Sample-level datasets: every item is one physical soil sample (all its photos)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phygrainnet.constants import NUM_POINTS, TARGET_COLUMNS
from phygrainnet.data.imaging import (
    canonical_image_cached,
    crop_at,
    geometric_aug,
    grid_centers,
    photometric_aug,
)

import cv2


@dataclass
class SampleRecord:
    sample_id: str
    images: list[np.ndarray]  # canonical-ppm RGB uint8
    target: np.ndarray | None = None  # (11,) cumulative %, may contain NaN
    mask: np.ndarray | None = None  # (11,) 1 = labelled
    upper_bound: float = float("nan")  # % passing finest external sieve
    source: str = "competition"
    cameras: list[str] = field(default_factory=list)

    @property
    def finest_index(self) -> int:
        if self.mask is None:
            return 0
        idx = np.where(self.mask > 0)[0]
        return int(idx[0]) if len(idx) else NUM_POINTS


def records_from_catalog(
    catalog: pd.DataFrame,
    labels: pd.DataFrame | None,
    target_ppm: float,
    border_frac: float,
    color_mode: str,
    cache_dir: str | Path | None,
    sample_ids: list[str] | None = None,
    source: str = "competition",
) -> list[SampleRecord]:
    """Build one SampleRecord per sample id. `labels` may carry mask_* columns and
    `passing_finest_sieve` (external data); without them every point is labelled."""
    lab = labels.set_index("sample_id") if labels is not None else None
    ids = sample_ids if sample_ids is not None else sorted(catalog["sample_id"].dropna().unique().tolist())
    records = []
    for sid in ids:
        rows = catalog[catalog["sample_id"] == sid]
        if rows.empty:
            raise ValueError(f"no photos for sample {sid}")
        imgs = [
            canonical_image_cached(r.path, float(r.ppm), target_ppm, border_frac, color_mode, cache_dir)
            for r in rows.itertuples()
        ]
        target = mask = None
        ub = float("nan")
        if lab is not None and sid in lab.index:
            row = lab.loc[sid]
            target = row[TARGET_COLUMNS].to_numpy(dtype=np.float64)
            mcols = [f"mask_{c}" for c in TARGET_COLUMNS]
            if all(c in lab.columns for c in mcols):
                mask = row[mcols].to_numpy(dtype=np.float64)
            else:
                mask = np.isfinite(target).astype(np.float64)
            if "passing_finest_sieve" in lab.columns:
                ub = float(row["passing_finest_sieve"])
        cams = rows["camera"].astype(str).tolist() if "camera" in rows else []
        records.append(SampleRecord(sid, imgs, target, mask, ub, source, cams))
    return records


def _to_tensor(tile: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.array(tile, copy=True)).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.25


def _sample_tile_stack(
    img: np.ndarray,
    fields_px: list[int],
    out_px: int,
    rng: np.random.Generator,
    scale_jitter: float,
    photometric: float,
    center: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    jitter = 1.0 + rng.uniform(-scale_jitter, scale_jitter) if scale_jitter > 0 else 1.0
    fields = [max(8, int(round(f * jitter))) for f in fields_px]
    big = max(fields)
    h, w = img.shape[:2]
    if center is None:
        cy = int(rng.integers(min(big // 2, h // 2), max(min(big // 2, h // 2) + 1, h - big // 2)))
        cx = int(rng.integers(min(big // 2, w // 2), max(min(big // 2, w // 2) + 1, w - big // 2)))
    else:
        cy, cx = center
    crop = crop_at(img, cy, cx, big)
    crop = geometric_aug(crop, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
    if photometric > 0:
        crop = photometric_aug(crop, rng, photometric)
    tiles = []
    for f in fields:
        o = (big - f) // 2
        t = crop[o : o + f, o : o + f]
        if f != out_px:
            t = cv2.resize(t, (out_px, out_px), interpolation=cv2.INTER_AREA if f > out_px else cv2.INTER_LINEAR)
        tiles.append(t)
    return tiles


class TrainTileDataset(Dataset):
    """Random multi-scale tiles per sample, with optional between-sample tile mixing.

    Tile mixing ("virtual soil blend"): with probability `mix_prob` a share lam of
    the tiles comes from sample A and the rest from sample B; the target becomes
    lam*F_A + (1-lam)*F_B (masks are AND-ed). This is a physically motivated
    augmentation for very few independent samples.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        fields_px: list[int],
        out_px: int,
        n_tiles: int,
        epoch_len: int,
        seed: int = 42,
        scale_jitter: float = 0.08,
        photometric: float = 1.0,
        mix_prob: float = 0.3,
    ) -> None:
        self.records = [r for r in records if r.target is not None]
        if not self.records:
            raise ValueError("no labelled records")
        self.fields_px, self.out_px, self.n_tiles = fields_px, out_px, n_tiles
        self.epoch_len = epoch_len
        self.seed = seed
        self.scale_jitter, self.photometric, self.mix_prob = scale_jitter, photometric, mix_prob
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_len

    def _tiles(self, rec: SampleRecord, n: int, rng: np.random.Generator) -> list[torch.Tensor]:
        out = []
        for _ in range(n):
            img = rec.images[int(rng.integers(0, len(rec.images)))]
            stack = _sample_tile_stack(img, self.fields_px, self.out_px, rng, self.scale_jitter, self.photometric)
            out.append(torch.stack([_to_tensor(t) for t in stack]))
        return out

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        a = self.records[int(rng.integers(0, len(self.records)))]
        if self.mix_prob > 0 and len(self.records) > 1 and rng.random() < self.mix_prob:
            b = self.records[int(rng.integers(0, len(self.records)))]
            lam = float(rng.uniform(0.25, 0.75))
            na = int(round(lam * self.n_tiles))
            tiles = self._tiles(a, na, rng) + self._tiles(b, self.n_tiles - na, rng)
            lam = na / self.n_tiles
            ta, tb = np.nan_to_num(a.target), np.nan_to_num(b.target)
            target = lam * ta + (1 - lam) * tb
            mask = a.mask * b.mask
            both = np.isfinite(a.upper_bound) and np.isfinite(b.upper_bound)
            ub = lam * a.upper_bound + (1 - lam) * b.upper_bound if both else np.nan
            finest = max(a.finest_index, b.finest_index)
        else:
            tiles = self._tiles(a, self.n_tiles, rng)
            target, mask, ub, finest = np.nan_to_num(a.target), a.mask, a.upper_bound, a.finest_index
        x = torch.stack(tiles)  # (T, S, 3, H, W)
        return {
            "x": x,
            "target": torch.tensor(target, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "upper_bound": torch.tensor(ub if np.isfinite(ub) else 100.0, dtype=torch.float32),
            "finest_index": torch.tensor(finest if np.isfinite(ub) else 0, dtype=torch.long),
        }


def inference_tiles(
    rec: SampleRecord,
    fields_px: list[int],
    out_px: int,
    max_tiles_per_image: int = 48,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic grid of multi-scale tiles over all photos (T, S, 3, H, W)."""
    rng = np.random.default_rng(seed)
    big = max(fields_px)
    step = min(fields_px)
    stacks = []
    for img in rec.images:
        h, w = img.shape[:2]
        centers = grid_centers(h, w, step, margin=min(big // 2, min(h, w) // 2))
        if len(centers) > max_tiles_per_image:
            idx = rng.choice(len(centers), size=max_tiles_per_image, replace=False)
            centers = [centers[k] for k in sorted(idx)]
        for c in centers:
            crop = crop_at(img, c[0], c[1], big)
            tiles = []
            for f in fields_px:
                o = (big - f) // 2
                t = crop[o : o + f, o : o + f]
                if f != out_px:
                    t = cv2.resize(t, (out_px, out_px), interpolation=cv2.INTER_AREA if f > out_px else cv2.INTER_LINEAR)
                tiles.append(_to_tensor(t))
            stacks.append(torch.stack(tiles))
    return torch.stack(stacks)


class UnlabeledTileDataset(Dataset):
    """Two augmented views of the same physical location for self-supervised pretraining."""

    def __init__(self, images: list[np.ndarray], field_px: list[int], out_px: int, epoch_len: int, seed: int = 42):
        self.images, self.field_px, self.out_px = images, field_px, out_px
        self.epoch_len, self.seed, self.epoch = epoch_len, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_len

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        img = self.images[int(rng.integers(0, len(self.images)))]
        f = int(self.field_px[int(rng.integers(0, len(self.field_px)))])
        h, w = img.shape[:2]
        half = min(f // 2 + f // 8, h // 2, w // 2)
        cy = int(rng.integers(half, max(half + 1, h - half)))
        cx = int(rng.integers(half, max(half + 1, w - half)))
        views = []
        for _ in range(2):
            off = rng.integers(-f // 8, f // 8 + 1, size=2)
            t = _sample_tile_stack(img, [f], self.out_px, rng, 0.1, 1.2, center=(cy + int(off[0]), cx + int(off[1])))[0]
            views.append(_to_tensor(t))
        return views[0], views[1]
