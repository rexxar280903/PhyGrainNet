# Metrics

## Primary metric (official, verified 2026-10-04)

```math
\mathrm{EMD}=\sum_{i=1}^{10}\left|F_i-\hat F_i\right|\cdot\left(\log_{10}x_{i+1}-\log_{10}x_i\right)
```

Mean over samples, range [0, 500]. Every interval weight is ≈ 0.5, so 1 percentage point of error at one point costs ≈ 0.5. The 200 mm value never contributes.

Implementation: `phygrainnet.metrics.kaggle_emd.emd_per_sample` / `competition_emd` (tested against hand-computed values). The training loss `masked_emd_loss` is the same quantity, restricted to labelled points.

## How we report CV

- OOF predictions from grouped folds (DEC-005), never from folds that saw the same site.
- `cv_emd` = mean over samples (the leaderboard definition); `cv_emd_group_mean` and `cv_emd_std_over_groups` show how much one site dominates.
- `worst_samples` and `per_point` in every `metrics.json` drive error analysis.

## Secondary diagnostics

Per-point contribution, interval-mass MAE, clay/silt/sand/gravel fraction error, D10/D30/D50/D60 error where defined, cross-view disagreement (std of per-photo predictions), monotonicity and endpoint violations (must be 0 by construction).
