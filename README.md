# PhyGrainNet

**Physical-scale-aware multi-view neural architecture for soil grain-size distribution estimation from phone photos — trained from scratch.**

PhyGrainNet is our entry to the Kaggle competition [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos) (BOKU / EU project GRID). The goal is a top private-leaderboard result with **models trained by us from random initialisation** — no ImageNet or other generic-image weights anywhere (DEC-001) — while using public soil datasets for our own pretraining (DEC-004).

## Project status

<!-- STATUS:START -->
**Overall project progress: 10%**  
`███░░░░░░░░░░░░░░░░░░░░░░░ 10%`

**Competition readiness: 10%**  
`███░░░░░░░░░░░░░░░░░░░░░░░ 10%`

**Current phase:** P2 — Dataset Audit (code ready, Kaggle run pending)  
**Next phase:** P5 — Baselines + first submission  
**Best grouped-CV EMD:** not established yet  
**Best public LB (not used for selection):** no submission yet  
**Deadline:** 2026-11-30 18:00 WIB
<!-- STATUS:END -->

`project_status.json` is the source of truth; `python scripts/update_readiness.py` regenerates this block (CI checks it). The week-by-week plan is in [`docs/SCHEDULE.md`](docs/SCHEDULE.md).

## The task in one table

| | |
|---|---|
| Input | 2–5 phone photos of one soil sample, camera at 21 cm, known px/mm per phone |
| Output | cumulative % passing at 11 diameters, 0.002 → 200 mm, non-decreasing, last value exactly 100 |
| Metric | log-weighted EMD: Σ\|F−F̂\|·Δlog10(d) over 10 intervals, mean over samples, lower is better |
| Train | **24 soils** (127 photos, Motorola Edge / Edge 60 Fusion / Samsung A52) |
| Test | **10 soils** from other sites (35 photos, **iPhone 14 / 16 only**); public LB = 3 fixed soils |

Reference scores on the training labels: training-median baseline ≈ 90–97, oracle nearest training curve ≈ 14.

## Approach

```text
photos ──► crop border ─► resample to 10 px/mm ─► gray-world colour
             │
             ├─► A-track: granulometry / Fourier / gradient / LBP features (mm units)
             │            └─► ridge / PLS in log-ratio space, kNN median
             │
             └─► PhyGrainNet-MV: tiles at 25.6 mm + 102.4 mm fields
                   GrainEncoder (RGB + fixed Sobel/Laplacian) ─► attention pooling over all tiles
                   ─► softmax interval masses ─► cumsum ─► valid curve
                   init: random │ own SimCLR (P000) │ own ETS-supervised model (P100)

OOF per model ─► pointwise-median ensemble (greedy on grouped CV) ─► make_valid ─► submission
```

Key choices and their reasons are in [`docs/DECISIONS.md`](docs/DECISIONS.md); architecture details in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start (Kaggle)

```bash
!git clone -q https://github.com/rexxar280903/PhyGrainNet.git /kaggle/working/PhyGrainNet
%cd /kaggle/working/PhyGrainNet
!pip install -q -e .
!python scripts/audit_dataset.py                                   # P2: catalog + checks
!python scripts/baseline_prior.py                                  # A000: first valid submission
!python scripts/train.py --config configs/classical/features_v1.yaml   # A001
!python scripts/train.py --config configs/phygrainnet/kaggle_scratch.yaml experiment_id=C100_smoke training.epochs=1 training.steps_per_epoch=2 cv.max_folds=1 inference.tta=1 inference.max_tiles_per_image=4  # GPU smoke test
!python scripts/train.py --config configs/phygrainnet/kaggle_scratch.yaml  # C100 full CV, after reviewing audit/smoke
!python scripts/ensemble.py --runs /kaggle/working/outputs/A001_ridge10 /kaggle/working/outputs/C100 \
    --out /kaggle/working/submission_H001.csv
```

Or import [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb). Full command list: [`docs/KAGGLE_WORKFLOW.md`](docs/KAGGLE_WORKFLOW.md).

The notebook defaults to audit, baseline, GPU smoke test and a verified results archive. Enable full CV and external pretraining explicitly after the first checks pass. Use a GitHub branch/commit containing the reviewed code. The latest local readiness audit is in [`docs/KAGGLE_READINESS_REVIEW.md`](docs/KAGGLE_READINESS_REVIEW.md).

## Experiment families

| ID | Meaning |
|---|---|
| A | hand-crafted physical features + small models |
| B | plain CNN baselines (random init) |
| C | PhyGrainNet, random init, competition data only |
| D | PhyGrainNet initialised from our own pretraining (SSL / ETS) |
| E / F / G | loss / ablation / TTA |
| H | ensembles |
| P | pretraining runs |

Every supervised run writes OOF/test curves, targets, groups, metrics, resolved configuration and environment details, plus a row in `/kaggle/working/registry.csv`; copy completed rows to `experiments/registry.csv`. Pretraining saves weights/config/history. Export results with `scripts/export_results.py`, persist a successful Kaggle version, and restore supervised checkpoints with `scripts/predict.py` to verify inference without retraining.

## Repository layout

```text
configs/            default.yaml + per-experiment configs (base: inheritance, CLI overrides)
docs/               decisions, rules audit, dataset facts, schedule, forum drafts, plan
notebooks/          kaggle_runner.ipynb
scripts/            audit, baseline, train (all pipelines), ensemble, submission, ETS tools
src/phygrainnet/
  data/             catalog (file names → sample/camera/ppm), imaging, labels + grouped folds, ETS labels
  features/         granulometry features
  models/           PhyGrainNet-MV, GrainBlock, constrained head, classical models
  training/         datasets, trainer, SimCLR, pipelines
  losses.py  metrics/  postprocess.py  submission.py  experiment.py
tests/              synthetic dataset with the real naming; end-to-end smoke tests
```

## Non-negotiable rules

1. No photo-level or sample-level random split: folds are by site group (DEC-005).
2. No weights that did not come from this repository (DEC-001; enforced when loading checkpoints).
3. Public leaderboard is never used for selection; no probing; no searching for test-site lab results (DEC-006).
4. Every submission maps to an experiment ID and a Git commit (`docs/LEADERBOARD_LOG.md`).
5. Every prediction passes `make_valid` and `validate_submission` before upload.
6. External data must be public, free and announced on the forum (DEC-004).

## Documentation

[Schedule](docs/SCHEDULE.md) · [Decisions](docs/DECISIONS.md) · [Competition rules](docs/COMPETITION_RULES.md) · [Dataset](docs/DATASET.md) · [External data](docs/EXTERNAL_DATA.md) · [Metrics](docs/METRICS.md) · [Architecture](docs/ARCHITECTURE.md) · [Project plan](docs/PROJECT_PLAN.md) · [Readiness gates](docs/READINESS_GATE.md) · [Research questions](docs/RESEARCH_QUESTIONS.md) · [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) · [Kaggle workflow](docs/KAGGLE_WORKFLOW.md) · [Forum drafts](docs/FORUM_POSTS.md) · [Leaderboard log](docs/LEADERBOARD_LOG.md) · [Paper plan](docs/PAPER_PLAN.md)
