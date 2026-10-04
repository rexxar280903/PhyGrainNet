# Kaggle Workflow

The dataset stays inside Kaggle. GitHub stores code, configs, documentation and the experiment registry.

## Session setup

Start from `notebooks/kaggle_runner.ipynb` (File → Import Notebook), then:

1. **Add Input → Competitions → soil-grain-size-from-photos** (and later the private dataset `ets-photogranulometry-10ppm`).
2. Settings: **Internet on** (for `git clone`), accelerator **GPU T4 x2** for neural runs (CPU is fine for A000/A001).
3. Run the setup cell:

```bash
!git clone -q https://github.com/rexxar280903/PhyGrainNet.git /kaggle/working/PhyGrainNet
%cd /kaggle/working/PhyGrainNet
!pip install -q -e .
```

## Commands

| Step | Command | Output |
|---|---|---|
| Audit | `python scripts/audit_dataset.py` | `/kaggle/working/phygrainnet/audit/` |
| A000 first submission | `python scripts/baseline_prior.py` | `submission_A000_median.csv` |
| A001 features | `python scripts/train.py --config configs/classical/features_v1.yaml` | `outputs/A001_*` |
| C100 scratch CNN | `python scripts/train.py --config configs/phygrainnet/mv_scratch.yaml` | `outputs/C100` |
| P000 SSL | `python scripts/train.py --config configs/pretrain/ssl.yaml` | `outputs/P000/ssl_encoder.pt` |
| P100 ETS | `python scripts/train.py --config configs/pretrain/ets.yaml` | `outputs/P100/ets_pretrained.pt` |
| D100 / D200 / D300 | `python scripts/train.py --config configs/phygrainnet/{mv_ssl,mv_ets,full}.yaml` | `outputs/D*` |
| Ensemble | `python scripts/ensemble.py --runs outputs/A001_ridge10 outputs/D200 --out submission_H001.csv` | validated CSV |
| Validate any CSV | `python scripts/validate.py submission.csv` | OK / INVALID |

Any config value can be overridden on the command line: `training.epochs=2 cv.max_folds=1 seed=43`.

## Keeping checkpoints between sessions

`/kaggle/working` is wiped when a session ends. Save the notebook with **Save Version → Save & Run All** so outputs persist, then add that notebook's output as an input to the next notebook (e.g. to reuse `P000/ssl_encoder.pt`). Point `training.init_checkpoint` at `/kaggle/input/<notebook-name>/outputs/P000/ssl_encoder.pt`.

## Submitting

Download the CSV from the notebook's output (or use **Submit to Competition** from the saved version), add the experiment ID and commit to the description, and log it in `docs/LEADERBOARD_LOG.md`.

## Research discipline

The public LB has 3 soils and has been probed; choose models by grouped-CV only (DEC-006).
