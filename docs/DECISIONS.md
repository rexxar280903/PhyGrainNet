# Decision Log

## DEC-001 — Primary architecture trained from scratch

Decision: the primary PhyGrainNet track uses no pretrained visual backbone.

Reason: the research contribution is custom architecture development rather than transfer learning.

Status: Accepted.

## DEC-002 — Kaggle as primary execution environment

Decision: dataset inspection, training, and inference run in Kaggle; raw competition data are not stored in GitHub.

Status: Accepted.

## DEC-003 — Grouped validation required

Decision: photographs sharing one physical soil target must remain in the same fold once the true grouping key is verified.

Reason: prevent target leakage across multiple views of the same material sample.

Status: Accepted.
