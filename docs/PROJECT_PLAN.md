# Project Plan

## Mission

PhyGrainNet is both a Kaggle competition project and a reproducible research project. The competition objective is to minimize the official Soil Grain Size from Photos metric. The research objective is to test whether physical-scale-aware, texture-aware, multi-view learning with a physically valid cumulative-distribution head outperforms simpler image-space models.

## Phase roadmap

| Phase | Deliverable | Completion threshold |
|---|---|---:|
| P0 | Problem, input/output, architecture hypothesis | 4% |
| P1 | Repository, configs, gates, experiment registry | 8% |
| P2 | Kaggle dataset audit | 14% |
| P3 | EDA and target/scale analysis | 20% |
| P4 | Exact metric + grouped CV + OOF | 28% |
| P5 | Statistical and CNN baselines | 38% |
| P6 | PhyGrainNet v0 | 50% |
| P7 | Physical multi-scale modelling | 60% |
| P8 | Texture + multi-view architecture | 70% |
| P9 | Controlled ablation study | 80% |
| P10 | Robust multi-seed evaluation | 87% |
| P11 | TTA/ensemble + Kaggle submission | 92% |
| P12 | Error analysis | 96% |
| P13 | Reproducibility packaging | 98% |
| P14 | Paper-ready evidence | 100% |

## Immediate sequence

1. Run the dataset audit directly in Kaggle.
2. Lock the real sample grouping key and file schema.
3. Reproduce the official metric exactly.
4. Build leakage-safe grouped folds.
5. Establish mean/median and simple-CNN baselines.
6. Train PhyGrainNet v0.
7. Add physical-scale, multi-scale, texture, and multi-view modules one at a time.
8. Run ablations, robust validation, inference, ensemble, and submissions.
9. Convert experiment evidence into a publication-ready research package.

## Success criteria

A good public leaderboard result is not sufficient. Final success requires a valid grouped evaluation protocol, zero invalid GSD predictions, traceable experiments, full ablations, reproducible Kaggle inference, and evidence that supports the proposed architectural contributions.
