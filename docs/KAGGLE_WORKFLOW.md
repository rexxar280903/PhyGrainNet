# Kaggle Workflow

The dataset stays inside Kaggle. GitHub stores code, configs, documentation and the experiment registry.

## Session setup

Start from `notebooks/kaggle_runner.ipynb` (File → Import Notebook), then:

1. **Add Input → Competitions → soil-grain-size-from-photos** (and later the private dataset `ets-photogranulometry-10ppm`).
2. Settings: **Internet on** (for `git clone` and dependencies), select an available GPU for neural runs (CPU is fine for A000/A001). The implementation uses one GPU; T4 x2 does not pool memory or run folds in parallel.
3. Use the reviewed branch/commit. Local changes must reach GitHub before cloning `main`; otherwise upload the reviewed repository snapshot. The notebook reuses an existing checkout and prints its commit; it does not silently pull updates.
4. Default **Run All** runs audit, saved A000 baselines, a small `C100_smoke` run and a verified archive. Full CV, classical models, SSL, ETS and ensemble are opt-in switches in the setup cell. With CPU only, set `RUN_SMOKE=False`.
5. Run the setup cell:

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
| C100 scratch CNN | `python scripts/train.py --config configs/phygrainnet/kaggle_scratch.yaml` | `outputs/C100` |
| P000 SSL | `python scripts/train.py --config configs/pretrain/ssl.yaml` | `outputs/P000/ssl_encoder.pt` |
| P100 ETS | `python scripts/train.py --config configs/pretrain/ets.yaml` | `outputs/P100/ets_pretrained.pt` |
| D100 / D200 / D300 | `python scripts/train.py --config configs/phygrainnet/{mv_ssl,mv_ets,full}.yaml` | `outputs/D*` |
| Ensemble | `python scripts/ensemble.py --runs outputs/A001_ridge10 outputs/D200 --out submission_H001.csv` | validated CSV |
| Validate any CSV | `python scripts/validate.py submission.csv` | OK / INVALID |
| Backup | `python scripts/export_results.py` | `/kaggle/working/phygrainnet_results.zip` with checksums |
| Restore inference | `python scripts/predict.py --checkpoints /kaggle/input/<saved-notebook>/outputs/C100/fold*.pt --out /kaggle/working/submission_C100_restored.csv` | validated CSV without retraining |

Any config value can be overridden on the command line: `training.epochs=2 cv.max_folds=1 seed=43`.

Use a distinct experiment ID for smoke tests, for example `experiment_id=C100_smoke`. Partial CV is recorded as `smoke_only` and rejected by the ensemble. Reusing an experiment ID replaces files in that folder; retain a saved version before rerunning it. The conservative Kaggle profile starts with batch size 2, six tiles per sample and inference chunk 16; actual memory/runtime must be measured on Kaggle.

For comparable baseline evaluation, A000/A001/C100 use five grouped folds and seed 42. LOGO remains available as a separate sensitivity run. Groups are inferred from IDs and still need domain/data review. SSL currently sees unlabelled validation and test photos; describe its CV as transductive, not strictly inductive.

## Keeping checkpoints between sessions

Treat `/kaggle/working` as temporary until a saved version has completed successfully. Save with **Save Version → Save & Run All**, with only the intended stages enabled, then verify its Output tab contains the artifacts. This runs a fresh copy; it is not merely a backup of the current interactive process. Add that saved notebook's output as input to the next notebook. Point `training.init_checkpoint` at `/kaggle/input/<notebook-name>/outputs/P000/ssl_encoder.pt`. See [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks) and [official output download commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-output).

The final notebook cell creates a ZIP containing outputs (including fold weights), audit, `/kaggle/working/registry.csv`, submissions and a source/config snapshot. A manifest records file hashes and Git state. Download it as a second backup. Raw competition images stay in Kaggle input; the canonical-image cache is rebuildable and excluded. The archive duplicates checkpoint storage, so check free disk space before exporting large pretraining runs.

Supervised runs store `oof.csv`, `test.csv`, `targets.csv`, `groups.csv`, `metrics.json`, `config.json` and `environment.json`. Metrics include per-sample, per-group and per-diameter errors, fold memberships and whether CV is complete. The notebook keeps command logs in `outputs/logs`. A000 now writes these evaluation files as well. SSL/ETS pretraining saves history/config with its weights rather than OOF files.

Checkpoints are saved after completed folds (or at the end of pretraining); they contain inference weights and configuration. They do **not** save optimizer/scheduler state for resuming an interrupted epoch. In a new session, run `scripts/predict.py` on saved supervised fold checkpoints and compare the resulting CSV with the original submission before declaring clean-session reproducibility.

## Submitting

Download the CSV from the notebook's output (or use **Submit to Competition** from the saved version), add the experiment ID and commit to the description, and log it in `docs/LEADERBOARD_LOG.md`.

## Research discipline

Choose models by complete grouped-CV only (DEC-006). The code being runnable does not establish model quality or pass the data/research readiness gates.
