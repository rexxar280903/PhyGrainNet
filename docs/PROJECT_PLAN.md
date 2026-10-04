# Project Plan

## Mission

Reach the top of the private leaderboard of *Predicting Soil Grain Size Distributions from Images* with models trained by us from random initialisation (DEC-001), using public soil data for pretraining where it helps (DEC-004), and keep every result reproducible enough to become a paper.

The week-by-week calendar, forum discussions and GitHub routine are in [`SCHEDULE.md`](SCHEDULE.md).

## Phase roadmap

| Phase | Deliverable | Code | Week | Cumulative |
|---|---|---|---|---:|
| P0 | Problem, I/O, hypothesis | docs | done | 4% |
| P1 | Repository, configs, gates, registry | repo | done | 8% |
| P2 | Kaggle dataset audit | `audit_dataset.py` | 1 | 14% |
| P3 | EDA, scale and label analysis | audit + notebook | 1 | 20% |
| P4 | Official metric + grouped CV + OOF | `metrics/`, `data/labels.py` | 1 | 28% |
| P5 | Baselines A000/A001, first submission | `baseline_prior.py`, A001 config | 1 | 38% |
| P6 | PhyGrainNet-MV from scratch (C100) | `models/`, `training/` | 2 | 50% |
| P7 | Own pretraining: SimCLR (P000), ETS (P100) | `training/ssl.py`, `pipelines.py` | 3–4 | 60% |
| P8 | Multi-scale / texture / multi-view ablations | configs | 4 | 70% |
| P9 | Full chain D300, multi-seed | `full.yaml` | 5 | 80% |
| P10 | Robustness: camera holdout, colour, TTA | configs | 6 | 87% |
| P11 | Ensemble + final submissions | `ensemble.py` | 5–8 | 92% |
| P12 | Error analysis | metrics.json | 3, 7 | 96% |
| P13 | Reproducibility packaging (clean-session rerun, tag) | repo | 7 | 98% |
| P14 | Paper-ready evidence | registry, docs | after | 100% |

## Expected score ranges (official metric)

| Stage | Expected grouped-CV EMD |
|---|---|
| training median / mean | 90–97 |
| A001 features | 40–60 (to be measured) |
| C100 scratch CNN | uncertain on 24 samples; may not beat A001 |
| D200/D300 with ETS pretraining | target < 25 |
| oracle nearest training curve | 14.2 (reference: perfect retrieval) |

## Success criteria

Valid submissions every week from week 1, honest grouped CV, zero invalid outputs, every submission traceable to an experiment ID and commit, and ablations that support each architectural claim.
