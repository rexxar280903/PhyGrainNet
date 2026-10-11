# Changelog

## 0.3.0 — 2026-10-10

- Indonesian offline interactive dashboard and methods paper, generated from repository status/config/profile.
- Partial MAC profiling, GPU environment capture and synthetic training benchmark.
- Complete grouped-OOF research reports, site-cluster bootstrap, fraction/diameter diagnostics and paired comparisons.
- Seven controlled ablation configs; one-tile gradient stability and empty-mask validation.
- Kaggle profiler/report steps and loader-inclusive training timing; archives include portal/docs.
- Scientific readiness remains evidence-based; software release completion is tracked separately.

## 0.2.0 — Competition pipeline (2026-10-04)

- Verified competition facts (metric, files, cameras, rules, deadline 2026-11-30 18:00 WIB) and recorded them in `docs/`.
- Revised DEC-001: all models from random init; own pretraining on public soil data allowed. Added DEC-004…DEC-012.
- Data: photo catalog for the real file names (umlauts, view suffixes, camera prefixes), effective PPM, canonical 10 px/mm images, colour normalisation, caching.
- Validation: site-grouped CV (13 groups), OOF outputs, registry rows, official EMD.
- Classical track: physically scaled granulometry / Fourier / gradient / LBP features; ridge & PLS in CLR space, kNN median.
- Neural track: PhyGrainNet-MV (multi-scale tiles, fixed texture filters, attention view pooling, constrained head), masked EMD loss, tile mixing, EMA, TTA; SimCLR pretraining; ETS supervised pretraining; checkpoint origin guard.
- External data: ETS label converter (11 points + masks + 80 µm bound) and image preparation script.
- Submission writer/validator, pointwise-median ensembling, Kaggle runner notebook, `update_readiness.py`.
- Schedule to the deadline in `docs/SCHEDULE.md`; forum post drafts in `docs/FORUM_POSTS.md`.

## 0.1.0 — Repository foundation

- Defined the competition and research missions.
- Added milestone-based progress and readiness gates.
- Adopted a Kaggle-first workflow without local dataset storage.
- Added initial PhyGrainNet v0 components and constrained GSD output.
- Added experiment registry, tests, and CI foundations.
