"""Regenerate notebooks/01_eda_first_submission.ipynb from this file (keeps the .ipynb diff-free).

    python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# PhyGrainNet — EDA + first submission (week 1: P2 → P5)

One notebook, top to bottom: **audit → EDA → A000 → A001 → best grouped-CV model → `/kaggle/working/submission.csv`**.

**Kaggle settings:** *Add Input → Competitions → soil-grain-size-from-photos*, **Internet on** (for `git clone`), accelerator **None (CPU)** is enough. Expect roughly 10–15 minutes on Kaggle's 4-core CPU, most of it the one-off feature extraction.

**Submitting:** *Save Version → Save & Run All*. When it finishes, open the version → *Output* → `submission.csv` → **Submit**. Paste the description printed by the last cell (experiment id + commit + CV) and add the same line to `docs/LEADERBOARD_LOG.md`.

Model choice uses grouped CV only; the public LB (3 soils) is never used to pick anything (DEC-006)."""),
    code("""import os

BRANCH = "main"                      # or a feature branch
REPO_URL = "https://github.com/rexxar280903/PhyGrainNet.git"
DATA_ROOT = os.environ.get("PGN_DATA_ROOT", "/kaggle/input/soil-grain-size-from-photos")
WORK = os.environ.get("PGN_WORK", "/kaggle/working")
REPO = os.environ.get("PGN_REPO", f"{WORK}/PhyGrainNet")
CLONE = os.environ.get("PGN_CLONE", "1") == "1"
RUN_CNN_SMOKE = os.environ.get("PGN_CNN_SMOKE", "0") == "1"   # True on a GPU session: 2-epoch, 1-fold C100 timing check

OUT = f"{WORK}/outputs"                  # oof.csv / test.csv / metrics.json per experiment (kept in the version output)
CACHE = "/tmp/phygrainnet_cache"         # canonical 10 px/mm images: 1–3 GB, not worth saving with the version
EDA_DIR = f"{WORK}/eda"
N_JOBS = os.cpu_count() or 2
OV = f"data.root={DATA_ROOT} output.dir={OUT} data.cache_dir={CACHE} data.n_jobs={N_JOBS}"
print(OV)"""),
    md("## 0. Setup"),
    code("""if CLONE:
    !rm -rf {REPO}
    !git clone -q -b {BRANCH} {REPO_URL} {REPO}
%cd {REPO}
!pip install -q -e . 2>&1 | tail -1
!git log --oneline -1"""),
    md("""## 1. Audit (P2 / gate G1)

Indexes every photo (sample id, phone, view, effective px/mm) and checks the labels. `problems` must be empty; a resize-factor warning means files are not at native size → check `data.scale_mode` (DEC-008)."""),
    code("""import json
import pandas as pd

!python scripts/audit_dataset.py --data-root {DATA_ROOT} --out-dir {WORK}/audit > {WORK}/audit.log 2>&1 || tail -20 {WORK}/audit.log
audit = json.load(open(f"{WORK}/audit/dataset_audit.json"))
print({k: audit[k] for k in ("n_train_samples", "n_test_samples", "n_photos")})
print("label checks:", {k: v for k, v in audit["labels"].items() if k != "column_means"})
print("PROBLEMS:", audit["problems"] or "none")
pd.DataFrame(audit["image_sizes"])"""),
    code("""pd.DataFrame(audit["field_of_view_mm"]).T.assign(ppm_min=pd.DataFrame(audit["ppm_by_camera"]).T["min"])"""),
    md("""## 2. EDA (P3)

`scripts/eda.py` writes everything to `/kaggle/working/eda/` (PNG, CSV, `eda_summary.json`, `eda_report.md`). It also computes the per-photo feature table that A001 needs and caches it, so section 4 starts immediately."""),
    code("""!python scripts/eda.py --data-root {DATA_ROOT} --out-dir {EDA_DIR} {OV} 2>&1 | grep -v "^\\[Parallel" | tail -40"""),
    code("""from IPython.display import Image, Markdown, display

def show(name):
    display(Image(f"{EDA_DIR}/{name}"))

display(Markdown(open(f"{EDA_DIR}/eda_report.md").read()))"""),
    md("""**Labels.** Two families (fine vs gravelly) and near-duplicate curves inside one site group — the reason folds are by site group (DEC-005). The right panel shows how unbalanced the fractions are."""),
    code("""show("01_label_curves.png")
pd.read_csv(f"{EDA_DIR}/label_table.csv").sort_values("D50")"""),
    md("""**Where an image-free guess loses points.** Per-sample EMD of the leave-one-group-out training mean, and its error per support point. Whatever the model, the 0.2–6.3 mm points are where most of the score is decided."""),
    code("""show("02_baseline_error.png")"""),
    md("""**Same physical field on every phone.** Each tile is 40 × 40 mm after resampling to 10 px/mm and gray-world colour. Check: grain sizes grow left→right/top→bottom with D50; the tray border is gone; test tiles look like the same kind of material."""),
    code("""show("03_train_thumbnails.png")"""),
    code("""show("04_test_thumbnails.png")"""),
    md("""**Camera shift (train: Motorola/Samsung, test: iPhone only).** Left: raw colour; middle: after gray-world (should overlap much more); right: texture energy at a common 2 px/mm — a camera whose box sits far from the others sharpens/denoises differently, which matters for fine-texture features."""),
    code("""show("05_colour_shift.png")
pd.read_csv(f"{EDA_DIR}/colour_table.csv").groupby(["split", "camera"])[["raw_L", "raw_a", "raw_b", "gw_a", "gw_b", "sharpness"]].median().round(2)"""),
    md("""**Feature space.** PCA fitted on training photos only (test photos projected). Triangles far outside the training cloud = extrapolation. Middle: features that move most on iPhone photos (orange = mostly within-soil noise, a candidate to drop). Right: features that track D50 / fines / gravel across the 24 soils."""),
    code("""show("06_feature_space.png")
pd.read_csv(f"{EDA_DIR}/test_nearest_train.csv")"""),
    md("""## 3. A000 — image-free safety net

Training mean and median with leave-one-site-group-out CV; writes `submission_A000.csv` from whichever has the lower grouped CV."""),
    code("""!python scripts/baseline_prior.py --data-root {DATA_ROOT} --out {WORK}/submission_A000.csv"""),
    md("""## 4. A001 — physically scaled features + small models (CPU)

Granulometry / Fourier / gradient / LBP / colour features in mm units → ridge & PLS in log-ratio space, kNN median; leave-one-group-out CV, OOF saved per model."""),
    code("""!python scripts/train.py --config configs/classical/features_v1.yaml {OV} 2>&1 | grep -v "^\\[Parallel" | tail -25"""),
    code("""import glob

rows = []
for mf in sorted(glob.glob(f"{OUT}/A001_*/metrics.json")):
    m = json.load(open(mf))
    worst = list(m["worst_samples"].items())[:3]
    rows.append({"experiment": m["experiment_id"], "cv_emd": m["cv_emd"], "cv_emd_median": m["cv_emd_median"],
                 "group_std": m["cv_emd_std_over_groups"], "worst": ", ".join(f"{k} {v:.0f}" for k, v in worst)})
cv_table = pd.DataFrame(rows).sort_values("cv_emd").reset_index(drop=True)
cv_table.round(2)"""),
    code("""import numpy as np
from matplotlib.figure import Figure

best_id = cv_table.experiment[0]
pts = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63"]
fig = Figure(figsize=(9, 3.4), layout="constrained")
ax = fig.subplots()
for k, eid in enumerate(dict.fromkeys([best_id, "A001_prior_median"])):
    pp = json.load(open(f"{OUT}/{eid}/metrics.json"))["per_point"]
    ax.bar(np.arange(10) + (k - 0.5) * 0.4, [pp[p] for p in pts], width=0.4, label=eid)
ax.set_xticks(range(10), pts); ax.set_xlabel("support point (mm)"); ax.set_ylabel("mean EMD contribution")
ax.set_title("grouped-CV error per point: best A001 model vs training median"); ax.legend()
fig.savefig(f"{EDA_DIR}/07_a001_per_point.png", dpi=110); show("07_a001_per_point.png")"""),
    md("""## 5. Submission file

The A001 model with the lowest grouped-CV EMD (it may legitimately be `prior_median` if no image model beats it yet — that is information, not a failure). The CSV is validated against the scorer's rules before it is written."""),
    code("""!python scripts/make_submission.py --runs {OUT}/A001_* --pick-best --data-root {DATA_ROOT} --out {WORK}/submission.csv
!python scripts/validate.py {WORK}/submission.csv --data-root {DATA_ROOT}
sub = pd.read_csv(f"{WORK}/submission.csv")
sub"""),
    code("""from phygrainnet.constants import DIAMETERS_MM, TARGET_COLUMNS

train_y = pd.read_csv(f"{DATA_ROOT}/Training_labels_updated.csv")
train_y.columns = ["sample_id", *TARGET_COLUMNS]
fig = Figure(figsize=(9, 4.2), layout="constrained")
ax = fig.subplots()
ax.fill_between(DIAMETERS_MM, train_y[TARGET_COLUMNS].min(), train_y[TARGET_COLUMNS].max(), color="0.88", label="training range")
for _, r in sub.iterrows():
    ax.plot(DIAMETERS_MM, r[TARGET_COLUMNS].to_numpy(float), lw=1.4, label=r.sample_id[:24])
ax.set_xscale("log"); ax.set_xlabel("diameter (mm)"); ax.set_ylabel("% passing")
ax.set_title(f"submitted test curves ({best_id})"); ax.legend(fontsize=6.5, ncol=2)
fig.savefig(f"{EDA_DIR}/08_submission_curves.png", dpi=110); show("08_submission_curves.png")"""),
    md("""## 6. Optional — C100 timing check (GPU session only)

Set `RUN_CNN_SMOKE = True` in the first cell on a **GPU T4 x2** session: 2 epochs on 1 fold of PhyGrainNet-MV from random init, to measure time per epoch before the full C100 run in week 2. Does not touch `submission.csv`."""),
    code("""import torch

if not RUN_CNN_SMOKE:
    print("skipped (RUN_CNN_SMOKE = False)")
elif not torch.cuda.is_available():
    print("skipped: no GPU in this session (the default batch needs a T4; on CPU it runs out of RAM)")
else:
    !python scripts/train.py --config configs/phygrainnet/mv_scratch.yaml {OV} training.epochs=2 cv.max_folds=1 experiment_id=C100_smoke output.registry={WORK}/smoke_registry.csv"""),
    md("""## 7. Log it

Paste the description into the Kaggle submission dialog and the table row into `docs/LEADERBOARD_LOG.md`; copy the new registry rows into `experiments/registry.csv` on GitHub (commit `exp(A001): ...`)."""),
    code("""import datetime as dt
import shutil

choice = json.load(open(f"{WORK}/submission.json"))
commit = !git rev-parse --short HEAD
print("Kaggle description:")
print(f"  {choice['experiment_id']} @ {commit[0]} | grouped-CV EMD {choice['cv_emd']:.2f} | from scratch, no pretrained weights")
print("\\nLEADERBOARD_LOG row:")
print(f"| S001 | {dt.date.today()} | {choice['experiment_id']} | {commit[0]} | {choice['cv_emd']:.2f} |  | first submission (A001 best of {len(cv_table)}) |")
shutil.copy("experiments/registry.csv", f"{WORK}/registry.csv")
pd.read_csv("experiments/registry.csv").tail(len(cv_table))"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("01_eda_first_submission.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
