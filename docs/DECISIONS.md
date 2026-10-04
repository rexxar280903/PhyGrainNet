# Decision Log

Goal fixed on 2026-10-04: **compete for the leaderboard with models trained by us from random initialisation, using additional public soil data where it helps.**

## DEC-001 — Every model starts from random initialisation (revised 2026-10-04)

Decision: no weights from generic-image pretraining (ImageNet, CLIP, DINO, timm, torchvision, ...) anywhere in the pipeline. Allowed: random init; our own self-supervised or supervised pretraining on soil photographs (competition photos, ETS dataset); fixed analytic filters (Sobel, Laplacian, morphology) and hand-crafted features.

Enforcement: `load_checkpoint` refuses any checkpoint whose `origin` is not `phygrainnet-from-scratch`; checkpoints record their `pretrain_data`.

Reason: the research contribution is the architecture and training recipe, and the claim "trained from scratch on soil data" must be auditable. External *data* is fine; external *weights* are not.

Status: Accepted (supersedes the 2026-10-03 wording "no pretrained visual backbone").

## DEC-002 — Kaggle as primary execution environment

Decision: dataset inspection, training, and inference run in Kaggle; raw competition data are not stored in GitHub. Large external data are uploaded as a private Kaggle Dataset after down-scaling.

Status: Accepted.

## DEC-003 — Grouped validation required

Decision: photographs sharing one physical soil target stay in the same fold; see DEC-005 for the grouping key.

Status: Accepted.

## DEC-004 — External data: ETS photogranulometry dataset

Decision: use Plante St-Cyr et al. (ÉTS Montréal), 321 samples / 12,714 photos, sieve PSD 80 µm–80 mm, CC BY 4.0, DOI 10.20383/103.01316. Labels are converted to the 11 competition points by log10-linear interpolation (same rule as the host); points below 80 µm are masked, and % passing 80 µm is used as an upper bound for finer points (`masked_emd_loss`).

Conditions: publicly and freely available (rule 2.6a), announced on the competition forum, no data that matches test sites.

Status: Accepted. Download pending (Globus only, >425 GB).

## DEC-005 — CV grouping key and fold scheme

Decision: `site_groups()` — same letter prefix and numeric id within 10 of the previous id → same group. On the 24 training ids this yields 13 groups (e.g. H366–H374 one group, H181/H183, H615–H617, H666/H668, H030–H038). Classical models: leave-one-group-out. Neural models: 5-fold grouped K-fold (compute budget).

Reason: H366–H374 are near-identical curves; random or sample-level splits would leak and make CV optimistic. Test ids (HPC_…) come from other sites, so site-level holdout is the closest proxy.

Status: Accepted; revisit if the host publishes site information.

## DEC-006 — Model selection uses grouped CV only; no leaderboard probing

Decision: the public LB (3 fixed test soils, already reverse-engineered by probing) is never used to select models or tune hyper-parameters. We do not probe it, and we do not search for published lab results of the test sites.

Reason: rules 4.b (no hand labelling / human prediction of test data), competition integrity, and 3 samples carry no statistical signal.

Status: Accepted.

## DEC-007 — Post-processing and ensembling

Decision: every prediction passes `make_valid` (clip to [0, 100], cumulative max, 200 mm column = exactly 100.0, 4 decimals). Ensembles use the pointwise (weighted) median, chosen greedily on OOF EMD.

Reason: the metric is a weighted L1 per support point, so the median is the loss-optimal combination, and the pointwise median of monotone curves is monotone.

Status: Accepted.

## DEC-008 — Canonical physical scale

Decision: every photo is resampled to 10 px/mm (0.1 mm/px) before features or tiles; PPM comes from `ppm_updated.csv` and is rescaled by the actual/native long side (`scale_mode: resize`). Tiles use concentric physical fields of 25.6 mm and 102.4 mm.

Reason: native PPM spans 11.49 (Motorola Edge) to 26.33 (Samsung A52); test phones are 13.94 / 19.53. 10 px/mm is below every camera so nothing is up-sampled. Verified on Kaggle that files are stored at native size (e.g. Motorola Edge 60 Fusion 2304×4096).

Status: Accepted; the P2 audit re-checks sizes for every file.

## DEC-009 — Colour normalisation for the camera shift

Decision: gray-world white balance per photo by default, plus photometric augmentation (white balance, exposure, contrast, saturation, blur, JPEG, noise). Lab standardisation is an ablation.

Reason: training photos come only from Motorola/Samsung, test photos only from iPhone 14/16.

Status: Accepted, to be validated by a camera-holdout CV (week 6).

## DEC-010 — Unlabelled test photos may be used for self-supervised pretraining

Decision: `configs/pretrain/ssl.yaml` includes test photos without any labels (no hand labelling, no pseudo-labels from humans).

Reason: allowed by the rules as written and it directly targets the iPhone domain gap. A forum question to the host is scheduled for week 1; if the host objects, set `ssl.include_test_photos: false` and re-run P000.

Status: Accepted, pending host confirmation.

## DEC-011 — Fixed training schedules, no early stopping on validation folds

Decision: epochs are fixed per config; validation folds are only evaluated, never used to stop or select checkpoints.

Reason: validation folds have 1–6 samples; early stopping on them would make OOF scores optimistic.

Status: Accepted.

## DEC-012 — Final submissions

Decision: Final 1 = best grouped-CV ensemble. Final 2 = a conservative blend (best ensemble shrunk towards the classical model / training median) that minimises the worst-group CV error.

Reason: 7 private soils; one catastrophic sample dominates the mean.

Status: Accepted.
