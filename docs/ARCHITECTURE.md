# PhyGrainNet Architecture

## Design principles

Compact, trained from random initialisation (DEC-001), and built around four properties of the task: physical scale, granular texture, several photos of one physical sample, and a valid cumulative output.

## Pipeline

```text
photos of one soil sample (2–5, any phone)
        │  PPM from ppm_updated.csv (rescaled if the file was resized)
        ▼
crop tray border → resample to 10 px/mm → gray-world colour      (DEC-008, DEC-009)
        │
        ▼
T tiles per sample, each with S concentric physical fields
(25.6 mm and 102.4 mm, both resized to 256 px)
        │
        ▼
GrainEncoder (shared across scales, + learned scale embedding)
  input: RGB + fixed Sobel magnitude + Laplacian  (texture stream)
  stem 5×5/2 → GrainBlock stages → mean ‖ std pooling
        │
        ▼
tile embedding (concat over scales → linear → LayerNorm)
        │
        ▼
AttentionPool over all tiles of all photos: mean ‖ std ‖ attention
        │
        ▼
MLP → ConstrainedDistributionHead: softmax over 11 interval masses ×100 → cumsum
        │
        ▼
valid GSD (monotone, ends at 100)  → make_valid → submission
```

`PhyGrainNetMV` (`src/phygrainnet/models/phygrainnet.py`), ≈ 1.5 M parameters at the default widths. `PhyGrainNetV0` is kept for the C000 ablation.

## GrainBlock

```text
input
 ├─ depthwise 3×3 ──────────┐
 └─ dilated depthwise 3×3 ──┤
                            concat → 1×1 conv → GroupNorm → GELU → channel gate → + input
```

GroupNorm because batches are small and samples few.

## Constrained distribution head

```math
m_i = 100\,\mathrm{softmax}(z)_i,\qquad \hat F_k=\sum_{i\le k} m_i,\qquad 0\le\hat F_1\le\dots\le\hat F_{11}=100 .
```

## Training recipe

| Element | Choice | Why |
|---|---|---|
| Loss | `masked_emd_loss` = official EMD on labelled points + hinge above the 80 µm bound for ETS samples | metric-aligned; handles partial external labels |
| Sampling | each item = one physical sample, T random tiles from random photos | sample-level supervision |
| Tile mixing | p = 0.3: tiles from two samples, target = λF_A + (1−λ)F_B | "virtual soil blends" from only 24 soils |
| Augmentation | rot90/flip, ±8 % scale jitter, white balance, exposure, contrast, saturation, blur, JPEG, noise | phone domain shift |
| Optimiser | AdamW, cosine schedule with warm-up, grad clip 1.0, EMA 0.995 | stable from scratch |
| Schedule | fixed epochs, no early stopping on validation (DEC-011) | honest OOF |
| Inference | grid tiles over every photo, 4-fold rotation TTA, mean of curves | uses all views |

## Pretraining stages (all from random init)

```text
P000  SimCLR on soil tiles (competition train+test photos, optionally ETS) ─┐
P100  supervised ETS (masked EMD)  ◄────────────────────────────────────────┘ encoder init
D200/D300  fine-tune on the 24 competition samples  ◄── P100 weights
```

## Classical track (A-series)

Per photo at 10 px/mm: Lab colour statistics, granulometric pattern spectra (morphological opening/closing with disks of 0.1–25.6 mm), radial Fourier energy per wavelength band (0.25–64 mm), scale-normalised gradient/Laplacian energy, multi-scale LBP. Averaged per sample, then ridge / PLS in centred-log-ratio space of the interval masses, or distance-weighted kNN median. These have no learned image weights and serve as a strong small-data baseline and ensemble member.

## Acceptance rule

A module stays only if a controlled ablation (same folds, same seeds) improves grouped-CV EMD, or if it enforces a necessary physical invariant.
