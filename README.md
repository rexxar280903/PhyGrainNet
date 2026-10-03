# PhyGrainNet

**Physical-Scale-Aware Multi-View Neural Architecture for Soil Grain-Size Distribution Estimation from Images**

PhyGrainNet is a research-first Kaggle project for the **Soil Grain Size from Photos** competition. The primary track develops a compact neural architecture **from scratch**, rather than fine-tuning an ImageNet backbone, while explicitly modelling physical image scale, soil texture, multi-scale context, multiple views of the same physical sample, and the monotonic structure of cumulative grain-size distributions (GSDs).

## Project status

**Overall project progress: 8%**

`██░░░░░░░░░░░░░░░░░░░░░░░░ 8%`

**Competition readiness: 3%**

`█░░░░░░░░░░░░░░░░░░░░░░░░░ 3%`

**Current phase:** P1 — Repository Foundation  
**Current readiness gate:** G0 — Research Specification  
**Best leakage-safe CV EMD:** not established yet  
**Best Kaggle score:** no submission yet  
**Primary execution environment:** Kaggle Notebook / Kaggle IDE  
**Default development seed:** 42

Progress is milestone-based, not manually estimated. `project_status.json` is the machine-readable source of truth and `scripts/update_readiness.py` regenerates the status block.

## Why we are doing this

The project has two simultaneous goals.

### Competition goal

Build a robust system that predicts the cumulative grain-size distribution of a soil sample from photographs and minimizes the competition's Earth Mover's Distance (EMD) evaluation metric without relying on leaderboard overfitting.

### Research goal

Investigate whether a purpose-built vision architecture can exploit information that generic image classifiers do not model explicitly:

- physical pixel-to-millimetre scale;
- fine-to-coarse granular texture;
- multiple photographs of the same physical soil sample;
- the fact that a valid GSD is monotonic and mass-conserving;
- metric-aligned learning using distribution-aware objectives.

The primary scientific output is intended to be a reproducible architecture and experimental record suitable for later manuscript development, not merely a Kaggle submission.

## Primary targets

The project will not declare success from a public leaderboard score alone. The current engineering targets are:

| Target | Goal |
|---|---:|
| Monotonicity violations | **0** |
| Invalid final cumulative endpoint | **0** |
| Reproducible Kaggle inference | **100%** |
| Leakage-safe CV established | Required before architecture claims |
| Initial CV milestone | EMD < 40 |
| Strong CV milestone | EMD < 25 |
| Competitive research milestone | EMD < 15 |
| Stretch milestone | EMD < 8 |
| Architecture evidence | Full ablation completed |
| Final scientific validation | Multiple seeds + robust grouped evaluation |

The numeric EMD milestones are engineering targets and may be revised after the exact competition metric and dataset structure are verified in the audit phase.

## Core hypothesis

A generic pixel-space CNN must learn physical scale, texture primitives, cross-view consistency, and valid cumulative-distribution geometry from a very small number of independent soil samples. PhyGrainNet instead injects these inductive biases directly into the design.

```text
Soil sample
    │
    ├── Photo 1 ─┐
    ├── Photo 2 ─┼─> physical-scale crops
    └── Photo N ─┘          │
                            v
                  multi-scale grain encoder
                            │
                  RGB + texture representation
                            │
                     scale aggregation
                            │
                      view aggregation
                            │
                  sample-level embedding
                            │
                    distribution head
                            │
                  11 non-negative masses
                            │
                      cumulative sum
                            │
                         valid GSD
```

## Input and output

The intended training unit is a **physical soil sample**, not an independent photograph.

Input:

```text
{image_1, image_2, ..., image_N} + physical-scale metadata
```

Output:

```text
[F(d1), F(d2), ..., F(d11)]
```

where `F(d)` is the cumulative percentage passing a grain diameter threshold. The distribution head predicts non-negative interval masses with a softmax and converts them to cumulative percentages using `cumsum`, guaranteeing monotonic predictions by construction.

## Kaggle-first data workflow

The competition dataset is **not stored in this repository** and does not need to be downloaded to the local computer. Training and dataset inspection are designed to run directly in Kaggle.

The default configuration expects a competition dataset under a Kaggle input mount such as:

```python
from pathlib import Path
DATA_ROOT = Path('/kaggle/input/soil-grain-size-from-photos')
```

The exact directory and filenames are verified by `scripts/audit_dataset.py` before training. No model training is considered valid until the dataset integrity gate passes.

See [`docs/KAGGLE_WORKFLOW.md`](docs/KAGGLE_WORKFLOW.md).

## Readiness gates

| Gate | Purpose | Initial status |
|---|---|---|
| G0 | Research specification | IN PROGRESS |
| G1 | Data integrity | BLOCKED until Kaggle audit |
| G2 | Experimental validity | BLOCKED |
| G3 | Architecture validation | BLOCKED |
| G4 | Competition ready | BLOCKED |
| G5 | Research ready | BLOCKED |

Full criteria: [`docs/READINESS_GATE.md`](docs/READINESS_GATE.md).

## Project phases

| Phase | Milestone | Cumulative progress |
|---|---|---:|
| P0 | Research specification | 4% |
| P1 | Repository foundation | 8% |
| P2 | Dataset audit | 14% |
| P3 | EDA | 20% |
| P4 | Exact metric + leakage-safe CV | 28% |
| P5 | Baselines | 38% |
| P6 | PhyGrainNet v0 | 50% |
| P7 | Physical multi-scale modelling | 60% |
| P8 | Texture + multi-view modelling | 70% |
| P9 | Ablation study | 80% |
| P10 | Robust evaluation | 87% |
| P11 | Ensemble + Kaggle submission | 92% |
| P12 | Error analysis | 96% |
| P13 | Reproducibility packaging | 98% |
| P14 | Paper-ready evidence | 100% |

Detailed plan: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Experiment families

```text
Axxx  non-neural/statistical baselines
Bxxx  simple CNN baselines from scratch
Cxxx  GrainBlock and physical-scale experiments
Dxxx  multi-view PhyGrainNet experiments
Exxx  loss-function experiments
Fxxx  ablations
Gxxx  test-time augmentation
Hxxx  ensembles
```

Every run must record its configuration, seed, fold, Git commit, validation metrics, runtime, and notes in `experiments/registry.csv`.

## Repository layout

```text
PhyGrainNet/
├── configs/              experiment configuration
├── docs/                 research protocol and decisions
├── notebooks/            EDA / analysis notebooks only
├── src/phygrainnet/      reusable source package
├── scripts/              Kaggle entry points
├── experiments/          experiment registry
├── outputs/              local/Kaggle run outputs (mostly ignored)
├── tests/                metric, model and data-invariant tests
├── project_status.json   machine-readable readiness state
└── .github/workflows/    CI and readiness automation
```

## Non-negotiable research rules

1. No photo-level random split when photographs share a physical soil target.
2. No architecture claim before leakage-safe grouped validation exists.
3. The primary architecture track uses no pretrained visual backbone.
4. Any optional pretrained model is labelled strictly as a comparison baseline.
5. Kaggle public leaderboard score never replaces local out-of-fold evaluation.
6. Every submission must map to an experiment ID and Git commit.
7. Predictions must satisfy physical distribution constraints before submission.
8. Changes that improve one fold but degrade generalisation are not automatically accepted.

## Immediate next milestone

**P2 — Dataset Audit in Kaggle**

The next run must establish the real dataset schema, unique sample IDs, photographs per sample, image dimensions, PPM metadata, missing values, duplicate risks, target monotonicity, train/test organisation, and potential leakage paths. Only then will the exact fold strategy and PhyGrainNet input pipeline be locked.

## Research documentation

- [Project plan](docs/PROJECT_PLAN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Research questions](docs/RESEARCH_QUESTIONS.md)
- [Readiness gates](docs/READINESS_GATE.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Dataset audit specification](docs/DATASET.md)
- [Metrics](docs/METRICS.md)
- [Kaggle workflow](docs/KAGGLE_WORKFLOW.md)
- [Competition rules audit](docs/COMPETITION_RULES.md)
- [Decision log](docs/DECISIONS.md)
- [Paper plan](docs/PAPER_PLAN.md)
- [Leaderboard/submission log](docs/LEADERBOARD_LOG.md)

---

**Current principle:** build the evaluation system first, then earn the right to optimize the model.
