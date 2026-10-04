# Experiment Registry

`registry.csv` gets one row per run from `scripts/train.py` (written on Kaggle; copy new rows back here and commit as `exp(<ID>): ...`).

| Family | Meaning | Configs |
|---|---|---|
| Axxx | non-neural / hand-crafted features | `configs/baseline/prior.yaml`, `configs/classical/features_v1.yaml` |
| Bxxx | plain CNN baselines from scratch | `configs/baseline/simple_cnn.yaml` |
| Cxxx | PhyGrainNet from random init, competition data only | `configs/phygrainnet/v0.yaml`, `mv_scratch.yaml` |
| Dxxx | PhyGrainNet initialised from our own pretraining | `mv_ssl.yaml`, `mv_ets.yaml`, `full.yaml` |
| Exxx | loss experiments | override `training.*` |
| Fxxx | ablations | copy a config, change one variable |
| Gxxx | test-time augmentation | override `inference.*` |
| Hxxx | ensembles | `scripts/ensemble.py` |
| Pxxx | pretraining runs (no competition labels) | `configs/pretrain/*.yaml` |

Every result used for model selection or a paper must be linked to a committed config and Git commit. `cv_emd` is the grouped out-of-fold official EMD.
