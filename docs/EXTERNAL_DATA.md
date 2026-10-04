# External Data

## ETS photogranulometry dataset (DEC-004)

- Plante St-Cyr, Duhaime, Dubé, Grenier (2025). *Dataset of soil images with corresponding particle size distributions for photogranulometry*. arXiv:2506.17469. Data: FRDR, DOI 10.20383/103.01316, **CC BY 4.0**.
- 321 samples (Montréal region, mostly silty sands with gravel), 12,714 photos, Canon EOS R5 + RF100 mm macro, 8192×5464 px at 39.4 µm/px (≈ 25.4 px/mm), moist and oven-dried states, white trays. > 425 GB.
- Labels: mechanical sieving (BNQ 2501-025/2013), % passing 14 sieves from 80 µm to 80 mm. Samples were washed to remove < 80 µm, so **no information below 0.08 mm** apart from the % passing 80 µm. (The FRDR landing page also mentions a hydrometer test; the paper does not. Check the Excel with `--inspect`.)

### What it can and cannot teach

| competition point (mm) | 0.002 · 0.0063 · 0.02 · 0.063 | 0.2 · 0.63 · 2 · 6.3 · 20 · 63 |
|---|---|---|
| ETS label | masked; P80 used as an upper bound | interpolated (round-trip error ≈ 1.5 EMD) |
| share of mean-baseline error on our 24 labels | 26 % | 74 % |

### Pipeline

1. Get a Globus account and transfer a subset (e.g. ≤ 6 photos per sample, both moisture states) from the FRDR Globus endpoint.
2. `python scripts/prepare_ets.py --src <soil_image> --out ets_canonical --inspect`, then without `--inspect` → images at 10 px/mm + `ets_catalog.csv` (relative paths).
3. `python scripts/convert_ets_labels.py --input <psd.xlsx> --inspect`, then `--output ets_canonical/ets_labels_11pt.csv --roundtrip-labels Training_labels_updated.csv`.
4. Upload `ets_canonical/` as a private Kaggle Dataset named `ets-photogranulometry-10ppm`.
5. `configs/pretrain/ets.yaml` (P100) → `configs/phygrainnet/mv_ets.yaml` (D200).

### Domain differences to handle

DSLR macro vs phone, white tray vs dark chamber, Canadian vs German/Austrian soils, moist/dry. Mitigations: canonical PPM, gray-world colour, photometric augmentation, moisture filter (`external.moisture`), fine-tune on competition data last.

## Candidate to check (not used yet)

*Dataset of close-range soil images and corresponding particle size distributions* (Data in Brief, PMC12152572). Must be checked for overlap with the GRID project / test sites before any use.
