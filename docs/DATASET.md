# Dataset

Facts verified on Kaggle on 2026-10-04. `scripts/audit_dataset.py` re-checks them on every run and writes `catalog.csv` + `dataset_audit.json`.

## Files

```text
/kaggle/input/soil-grain-size-from-photos/
├── Training_labels_updated.csv          24 rows: sample_id + 11 cumulative % columns
├── ppm_updated.csv                      phone, camera, width, height, ppm
├── sample_submission.csv                10 test ids, placeholder zeros
├── Training-All_Photos_updated/Training-All_Photos_updated/   127 photos
└── Test_All_Photos/Test_All_Photos/                           35 photos
```

## Naming

- Train: `Motorola_Edge_H030_01.jpg`, `Motorola_Edge_60_fusion_H374_01.jpg` (camera prefix + sample id + view).
- Test: `iPhone14_HPC_Münster_BS6_9,0-10m (1).JPG`, `iPhone16_HPC_Airbus BS10-4bis7 (2).JPG`.
- The submission id for the Münster sample is `HPC_Muenster_BS6_9_0-10m`; matching therefore uses `norm_key()` (umlauts transliterated, punctuation removed). Ids that end in digits (`Kleinkummerfeld 2-2`) are matched on the full stem first.

## Cameras and scale (`ppm_updated.csv`)

| phone | native size | px/mm | split |
|---|---|---:|---|
| Motorola Edge (edge 20) | 4000×1800 | 11.492 | train |
| Motorola Edge 60 Fusion | 4096×2304 | 12.465 | train (H374 re-shoot) |
| Samsung A52 (SM-A525F) | 9248×6936 | 26.330 | train |
| iPhone 14 | 4032×3024 | 13.942 | test only |
| iPhone 16 | 5712×4284 | 19.525 | test only |

Camera-to-soil distance is always 21 cm, same lighting (host). Files are stored at native resolution (checked: Motorola Edge 60 Fusion 2304×4096 portrait), so the field of view is roughly 29–35 cm wide. Photos show the tray edge → `border_frac: 0.06`.

## Labels

- Sieve + sedimentation analyses, interpolated linearly on log10(d) to the 11 DIN EN ISO 14688-1 points by the host.
- 24 training samples; two broad families: fine soils (silt/sand, gravel ≈ 0 %) and gravelly soils (gravel 47–77 %).
- H374 was re-photographed with the Motorola Edge 60 Fusion and its labels updated; H031 was removed (soil no longer available).
- Near-duplicate curves: H366/H367/H368/H371/H372/H374, H181/H183, H368/H615 → handled by `site_groups()` (DEC-005).

## Test set

10 samples (`HPC_*`: Airbus ×3, Audorfring, Münster BS6 9–10 m, Kleinkummerfeld ×4, Testfeld Lidl WHV), 2–5 photos each, iPhone only, from sites that do not appear in training. Public LB = 3 fixed soils.

## Reference numbers (official metric, 24 training labels)

| predictor | EMD |
|---|---:|
| host equal-mass curve | 102.0 |
| training mean, leave-one-out | 89.9 |
| training median, leave-one-out | 97.3 |
| oracle 1-NN (always picks the closest training curve) | 14.2 |

The four fine points (≤ 0.063 mm) carry about 26 % of the mean-baseline error; 0.2–6.3 mm carry about 70 %.
