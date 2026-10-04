# Research Questions and Hypotheses

## Central question

Can a compact architecture designed around physical scale, granular texture, multi-view observation, and valid distribution geometry estimate soil grain-size distributions more accurately and robustly than simpler image-space models under the same leakage-safe protocol?

## RQ1 — Physical scale

Does PPM-aware physical normalization improve GSD estimation relative to conventional pixel-space resizing?

**H1:** lower grouped-CV EMD for a matched physical-scale-aware model.

## RQ2 — Multi-scale perception

Do multiple physical fields of view improve joint estimation of fine and coarse fractions?

**H2:** multi-scale inputs reduce EMD and per-diameter MAE.

## RQ3 — Multi-view representation

Does aggregating photographs from the same physical sample outperform single-image prediction?

**H3:** sample-level aggregation lowers EMD and prediction variance across views.

## RQ4 — Constrained output

Does predicting non-negative interval masses followed by cumulative summation improve validity and accuracy versus independent cumulative regression?

**H4:** zero monotonicity violations with equal or better EMD.

## RQ5 — Texture inductive bias

Does an explicit texture/gradient stream add information beyond RGB-only encoding?

**H5:** improved fine/intermediate fraction prediction without unacceptable fold instability.

## RQ6 — Metric-aligned learning

Does an EMD-aligned objective improve the official validation score relative to MAE/MSE-only objectives?

**H6:** lower official EMD under the same folds and architecture.

## RQ7 — Pretraining on external soil data from scratch

Does supervised pretraining on a public soil dataset with partial labels (ETS, 0.08–80 mm) improve the competition score of a randomly initialised model, including on points the external data never labels?

**H7:** D200 < C100 in grouped-CV EMD, with gains concentrated at 0.2–20 mm.

## RQ8 — Self-supervised pretraining under a camera shift

Does SimCLR on unlabelled soil tiles (including the iPhone test photos) reduce the train-camera → test-camera gap?

**H8:** D100 < C100, and a smaller gap in camera-holdout CV.

## Evidence rule

A hypothesis is supported only by controlled comparisons using the same folds, preprocessing, seed policy, and metric implementation, with improvement not isolated to a single fold.
