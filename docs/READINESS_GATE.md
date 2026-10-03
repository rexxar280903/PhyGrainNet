# Readiness Gates

Progress measures how much work has been completed. A gate determines whether results are scientifically safe to promote to the next stage.

## G0 — Research Specification

- [x] Prediction task defined.
- [x] Input/output concept defined.
- [x] Primary architecture hypothesis documented.
- [x] Research questions documented.
- [x] Experiment ID convention defined.
- [x] Kaggle selected as primary execution environment.
- [ ] Live competition files, rules, and official metric implementation verified.

Status: **IN PROGRESS**

## G1 — Data Integrity

No serious model training is accepted before all items pass.

- [ ] Train and test images indexed.
- [ ] Physical sample IDs identified.
- [ ] Photos per sample verified.
- [ ] PPM/scale metadata verified.
- [ ] Missing metadata checked.
- [ ] Duplicate/near-duplicate risk checked.
- [ ] Repeated targets checked.
- [ ] Target columns/order/bounds/monotonicity verified.
- [ ] Train/test schema confirmed.
- [ ] Leakage paths documented.

Status: **BLOCKED**

## G2 — Experimental Validity

- [ ] Official Kaggle metric reproduced exactly.
- [ ] Metric unit tests pass.
- [ ] Grouped split implemented.
- [ ] Zero physical-sample overlap across train/validation.
- [ ] OOF predictions implemented.
- [ ] Mean/median baseline established.
- [ ] Simple neural baseline established.
- [ ] Fold statistics recorded.

Status: **BLOCKED**

## G3 — Architecture Validation

- [ ] PhyGrainNet trains without NaN/Inf.
- [ ] Zero monotonicity violations.
- [ ] Endpoint equals 100 by construction.
- [ ] Beats statistical baseline.
- [ ] Beats matched simple CNN.
- [ ] Improvement appears in the majority of folds.
- [ ] Physical-scale, multi-scale, texture, multi-view and head ablations complete.
- [ ] Finalist tested with multiple seeds.

Status: **BLOCKED**

## G4 — Competition Ready

- [ ] Final CV completed.
- [ ] TTA and ensemble validated rather than assumed.
- [ ] Submission schema and row mapping verified.
- [ ] All outputs valid.
- [ ] Clean-session Kaggle inference is reproducible.
- [ ] Every submission maps to experiment ID and Git commit.
- [ ] CV/public-leaderboard discrepancy reviewed.

Status: **BLOCKED**

## G5 — Research Ready

- [ ] Research questions answered from controlled evidence.
- [ ] Baseline and ablation tables complete.
- [ ] Multi-seed uncertainty reported.
- [ ] Geotechnical secondary metrics reported.
- [ ] Error analysis and limitations complete.
- [ ] Architecture diagram finalized.
- [ ] Reproducibility instructions verified.
- [ ] Paper-ready tables/figures and evidence map prepared.

Status: **BLOCKED**
