"""P3 exploratory data analysis.

Three questions drive everything here:

1. What do the 24 training curves look like, and which diameters carry the error of
   an image-free baseline? (labels)
2. Do the photos look comparable once they are at the same physical scale?
   (thumbnails at 10 px/mm, photo counts per camera)
3. How far do the iPhone test photos sit from the Motorola/Samsung training photos,
   in colour, sharpness and feature space? (domain shift — DEC-009)

Figures are built with `matplotlib.figure.Figure` (no pyplot state), saved as PNG and
summarised in `eda_summary.json` + `eda_report.md`. Nothing here reads test labels
(there are none) or tunes anything on the test photos.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageOps

from phygrainnet.constants import DIAMETERS_MM, TARGET_COLUMNS
from phygrainnet.data.imaging import canonical_image_cached, crop_border, normalize_color
from phygrainnet.data.labels import label_matrix, leave_one_group_out
from phygrainnet.metrics.kaggle_emd import emd_per_sample, per_point_contribution

LOG_D = np.log10(DIAMETERS_MM)
INTERVALS = [f"{a:g}–{b:g}" for a, b in zip([0.0, *DIAMETERS_MM[:-1]], DIAMETERS_MM)]
META_COLS = {"path", "file", "split", "sample_id", "camera"}
TRAIN_MARK, TEST_MARK = "o", "^"


# ----------------------------------------------------------------------------- labels

def d_value(curve: np.ndarray, p: float) -> float:
    """Diameter (mm) at which `p` % passes, interpolated linearly in log10(d).
    NaN when more than p % is already finer than 2 µm."""
    c = np.asarray(curve, dtype=np.float64)
    if c[0] >= p:
        return float("nan")
    i = int(np.argmax(c >= p))
    lo, hi = c[i - 1], c[i]
    t = 0.0 if hi == lo else (p - lo) / (hi - lo)
    return float(10 ** (LOG_D[i - 1] + t * (LOG_D[i] - LOG_D[i - 1])))


def label_table(labels: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    """Per-sample soil fractions and characteristic diameters (DIN EN ISO 14688-1 bounds)."""
    y = label_matrix(labels)
    col = {c: i for i, c in enumerate(TARGET_COLUMNS)}
    out = pd.DataFrame({"sample_id": labels["sample_id"], "group": groups})
    out["clay"] = y[:, col["0.002"]]
    out["silt"] = y[:, col["0.063"]] - y[:, col["0.002"]]
    out["sand"] = y[:, col["2"]] - y[:, col["0.063"]]
    out["gravel"] = y[:, col["63"]] - y[:, col["2"]]
    out["cobbles"] = 100 - y[:, col["63"]]
    out["fines"] = y[:, col["0.063"]]
    for p in (10, 30, 50, 60):
        out[f"D{p}"] = [d_value(r, p) for r in y]
    out["Cu"] = out["D60"] / out["D10"]
    out["family"] = np.where(out["gravel"] + out["cobbles"] >= 30, "gravelly", "fine")
    return out.round(4)


def loo_baselines(y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Leave-one-group-out predictions of the image-free baselines (no images used)."""
    preds = {}
    for name, fn in (("mean", lambda a: a.mean(0)), ("median", lambda a: np.median(a, 0))):
        oof = np.zeros_like(y)
        for trn, val in leave_one_group_out(groups):
            oof[val] = fn(y[trn])
        preds[name] = oof
    return preds


def fig_label_curves(labels: pd.DataFrame, ltab: pd.DataFrame) -> Figure:
    y = label_matrix(labels)
    fig = Figure(figsize=(14, 5.2), layout="constrained")
    ax, ax2 = fig.subplots(1, 2, width_ratios=[1, 1.3])
    cmap = matplotlib.colormaps["tab20"]
    for i, (sid, g, fam) in enumerate(zip(ltab.sample_id, ltab.group, ltab.family)):
        ax.plot(DIAMETERS_MM, y[i], color=cmap(g % 20), lw=1.4, ls="-" if fam == "gravelly" else "--", alpha=0.9)
    ax.plot(DIAMETERS_MM, np.median(y, 0), color="black", lw=2.5, label="training median")
    ax.set_xscale("log")
    ax.set_xlabel("diameter (mm)")
    ax.set_ylabel("% passing")
    ax.set_title(f"{len(y)} training curves — colour = CV site group, dashed = fine family")
    for x in (0.002, 0.063, 2, 63):
        ax.axvline(x, color="0.8", lw=0.8, zorder=0)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25, which="both")

    order = ltab.sort_values("D50", na_position="first").index
    fr = ltab.loc[order, ["clay", "silt", "sand", "gravel", "cobbles"]]
    bottom = np.zeros(len(fr))
    colours = ["#7b5e3b", "#b08d57", "#e2c48e", "#8c8c8c", "#4d4d4d"]
    for c, colr in zip(fr.columns, colours):
        ax2.bar(range(len(fr)), fr[c], bottom=bottom, color=colr, label=c, width=0.85)
        bottom += fr[c].to_numpy()
    ax2.set_xticks(range(len(fr)), [f"{s}\n(g{g})" for s, g in zip(ltab.loc[order, "sample_id"], ltab.loc[order, "group"])],
                   rotation=90, fontsize=7)
    ax2.set_ylabel("mass %")
    ax2.set_title("soil fractions, samples sorted by D50")
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    return fig


def fig_baseline_error(y: np.ndarray, ids: list[str], preds: dict[str, np.ndarray]) -> Figure:
    fig = Figure(figsize=(14, 4.4), layout="constrained")
    ax, ax2 = fig.subplots(1, 2)
    per = emd_per_sample(y, preds["mean"])
    order = np.argsort(-per)
    ax.bar(range(len(per)), per[order], color="#5b7fa6")
    ax.axhline(per.mean(), color="black", lw=1, ls="--", label=f"mean {per.mean():.1f}")
    ax.set_xticks(range(len(per)), [ids[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("EMD")
    ax.set_title("per-sample EMD of the leave-one-group-out training mean")
    ax.legend()
    w = 0.38
    x = np.arange(10)
    for k, (name, p) in enumerate(preds.items()):
        pp = per_point_contribution(y, p)
        ax2.bar(x + (k - 0.5) * w, pp, width=w, label=f"LOGO {name} (EMD {emd_per_sample(y, p).mean():.1f})")
    ax2.set_xticks(x, TARGET_COLUMNS[:-1])
    ax2.set_xlabel("support point (mm)")
    ax2.set_ylabel("mean EMD contribution")
    ax2.set_title("where the error of an image-free guess sits")
    ax2.legend()
    return fig


# ----------------------------------------------------------------------------- photos

def photo_counts(catalog: pd.DataFrame) -> pd.DataFrame:
    return catalog.groupby(["split", "camera"]).agg(photos=("file", "size"), samples=("sample_id", "nunique")).reset_index()


def fig_thumbnails(catalog: pd.DataFrame, order: list[str], titles: dict[str, str], cfg: dict,
                   field_mm: float = 40.0, n_cols: int = 6, title: str = "") -> Figure:
    """One centre crop of `field_mm` × `field_mm` per sample at the canonical scale —
    every tile shows the same physical area whatever the phone."""
    d = cfg["data"]
    tp = float(d["target_ppm"])
    side = int(round(field_mm * tp))
    n_rows = int(np.ceil(len(order) / n_cols))
    fig = Figure(figsize=(2.3 * n_cols, 2.55 * n_rows), layout="constrained")
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols)).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, sid in zip(axes, order):
        r = catalog[catalog.sample_id == sid].iloc[0]
        img = canonical_image_cached(r.path, float(r.ppm), tp, float(d.get("border_frac", 0.06)),
                                     d.get("color_mode", "grayworld"), d.get("cache_dir"))
        h, w = img.shape[:2]
        s = min(side, h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        ax.imshow(img[y0 : y0 + s, x0 : x0 + s])
        ax.set_title(titles.get(sid, sid), fontsize=7.5)
    fig.suptitle(title or f"centre {field_mm:g} mm × {field_mm:g} mm at {tp:g} px/mm ({d.get('color_mode')})", fontsize=11)
    return fig


def _small_rgb(path: str, max_side: int) -> np.ndarray:
    """Fast low-resolution decode (JPEG draft mode) for colour statistics."""
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return np.asarray(im.convert("RGB"))


def colour_table(catalog: pd.DataFrame, border_frac: float = 0.06, max_side: int = 768,
                 sharp_ppm: float = 2.0) -> pd.DataFrame:
    """Per-photo colour and sharpness statistics, raw and after gray-world.

    Sharpness = variance of the Laplacian after resampling to `sharp_ppm` px/mm,
    so it compares cameras at the same physical scale (texture at ≥ 1 mm).
    """
    rows = []
    for r in catalog.itertuples():
        img = crop_border(_small_rgb(r.path, max_side), border_frac)
        rec = {"file": r.file, "split": r.split, "sample_id": r.sample_id, "camera": r.camera}
        for tag, im in (("raw", img), ("gw", normalize_color(img, "grayworld"))):
            lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            rec.update({f"{tag}_L": lab[:, 0].mean(), f"{tag}_a": lab[:, 1].mean() - 128, f"{tag}_b": lab[:, 2].mean() - 128,
                        f"{tag}_Lstd": lab[:, 0].std()})
        # long sides, so EXIF-rotated files (common on iPhones) are handled
        ppm_small = float(r.ppm) * max(img.shape[:2]) / ((1 - 2 * border_frac) * max(float(r.width), float(r.height)))
        lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = sharp_ppm / ppm_small
        if f < 1:
            lum = cv2.resize(lum, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        rec["sharpness"] = float(cv2.Laplacian(lum, cv2.CV_32F).var())
        rows.append(rec)
    return pd.DataFrame(rows)


def fig_colour_shift(ct: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(15, 4.6), layout="constrained")
    axes = fig.subplots(1, 3)
    cams = sorted(ct.camera.dropna().unique())
    cmap = matplotlib.colormaps["tab10"]
    for k, (tag, label) in enumerate((("raw", "raw photo"), ("gw", "after gray-world"))):
        ax = axes[k]
        for i, cam in enumerate(cams):
            s = ct[ct.camera == cam]
            mk = TEST_MARK if (s.split == "test").all() else TRAIN_MARK
            ax.scatter(s[f"{tag}_a"], s[f"{tag}_b"], s=18 + s[f"{tag}_L"] / 4, color=cmap(i), marker=mk, alpha=0.75, label=cam)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_xlabel("mean a* (green ← → red)")
        ax.set_ylabel("mean b* (blue ← → yellow)")
        ax.set_title(f"photo colour, {label} (size ∝ L*)")
    axes[0].legend(fontsize=7)
    ax = axes[2]
    data = [np.log10(ct.loc[ct.camera == c, "sharpness"] + 1e-6) for c in cams]
    ax.boxplot(data, tick_labels=[c.replace(" ", "\n", 1) for c in cams])
    ax.set_ylabel("log10 Laplacian variance @ 2 px/mm")
    ax.set_title("texture energy at a common physical scale")
    return fig


# ----------------------------------------------------------------------------- features

def feature_columns(feats: pd.DataFrame) -> list[str]:
    cols = [c for c in feats.columns if c not in META_COLS]
    x = feats[cols].to_numpy(dtype=np.float64)
    keep = np.isfinite(x).all(0) & (x.std(0) > 1e-9)
    return [c for c, k in zip(cols, keep) if k]


def standardised(feats: pd.DataFrame, cols: list[str]) -> np.ndarray:
    tr = feats[feats.split == "train"][cols].to_numpy(dtype=np.float64)
    mu, sd = tr.mean(0), tr.std(0) + 1e-12
    return (feats[cols].to_numpy(dtype=np.float64) - mu) / sd


def feature_shift(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per feature: test-vs-train shift in train-SD units, and how much of the
    training variance is between samples (high = stable across views of one soil)."""
    z = pd.DataFrame(standardised(feats, cols), columns=cols)
    z["split"] = feats["split"].to_numpy()
    z["sample_id"] = feats["sample_id"].to_numpy()
    tr = z[z.split == "train"]
    shift = z[z.split == "test"][cols].mean() - tr[cols].mean()
    within = tr.groupby("sample_id")[cols].var(ddof=0).mean()
    total = tr[cols].var(ddof=0) + 1e-12
    out = pd.DataFrame({"feature": cols, "test_shift_sd": shift.to_numpy(),
                        "between_sample_share": (1 - within / total).clip(0, 1).to_numpy()})
    return out.sort_values("test_shift_sd", key=np.abs, ascending=False).reset_index(drop=True)


def feature_label_corr(feats: pd.DataFrame, cols: list[str], ltab: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of sample-mean features with log D50, fines and gravel."""
    m = feats[feats.split == "train"].groupby("sample_id")[cols].mean()
    t = ltab.set_index("sample_id").loc[m.index]
    targets = {"log10_D50": np.log10(t["D50"]), "fines": t["fines"], "gravel": t["gravel"] + t["cobbles"]}
    rank_f = m.rank()
    out = {}
    for name, v in targets.items():
        rv = pd.Series(v, index=m.index).rank()
        out[name] = rank_f.corrwith(rv)
    df = pd.DataFrame(out)
    df["max_abs"] = df.abs().max(axis=1)
    return df.sort_values("max_abs", ascending=False)


def fig_feature_space(feats: pd.DataFrame, cols: list[str], shift: pd.DataFrame, corr: pd.DataFrame,
                      top: int = 15) -> Figure:
    z = standardised(feats, cols)
    tr = (feats.split == "train").to_numpy()
    # PCA fitted on training photos only; test photos are projected.
    mu = z[tr].mean(0)
    _, _, vt = np.linalg.svd(z[tr] - mu, full_matrices=False)
    pc = (z - mu) @ vt[:2].T
    var = (((z[tr] - mu) @ vt[:2].T).var(0) / (z[tr] - mu).var(0).sum())
    fig = Figure(figsize=(16, 5.2), layout="constrained")
    ax, ax2, ax3 = fig.subplots(1, 3, width_ratios=[1.2, 1, 1])
    cams = sorted(feats.camera.dropna().unique())
    cmap = matplotlib.colormaps["tab10"]
    for i, cam in enumerate(cams):
        m = (feats.camera == cam).to_numpy()
        mk = TEST_MARK if not tr[m].any() else TRAIN_MARK
        ax.scatter(pc[m, 0], pc[m, 1], color=cmap(i), marker=mk, s=26, alpha=0.8, label=cam)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of train variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_title("photo-level features, PCA fitted on train (▲ = test)")
    ax.legend(fontsize=7)

    s = shift.head(top).iloc[::-1]
    ax2.barh(s.feature, s.test_shift_sd, color=np.where(s.between_sample_share > 0.5, "#5b7fa6", "#c98b5b"))
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("test mean − train mean (train SD)")
    ax2.set_title("largest train→test shifts\n(blue: mostly between-soil variance)")
    ax2.tick_params(axis="y", labelsize=7)

    c = corr.head(top).iloc[::-1]
    y = np.arange(len(c))
    for k, col in enumerate(["log10_D50", "fines", "gravel"]):
        ax3.barh(y + (k - 1) * 0.27, c[col], height=0.27, label=col)
    ax3.set_yticks(y, c.index, fontsize=7)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("Spearman ρ (sample means, 24 soils)")
    ax3.set_title("features most related to the labels")
    ax3.legend(fontsize=7)
    return fig


def nearest_training(feats: pd.DataFrame, cols: list[str], k: int = 3) -> pd.DataFrame:
    """For each test soil, the k closest training soils in standardised feature space."""
    z = pd.DataFrame(standardised(feats, cols), columns=cols)
    z["sample_id"] = feats["sample_id"].to_numpy()
    z["split"] = feats["split"].to_numpy()
    tr = z[z.split == "train"].groupby("sample_id")[cols].mean()
    te = z[z.split == "test"].groupby("sample_id")[cols].mean()
    rows = []
    for sid, v in te.iterrows():
        d = np.sqrt(((tr - v) ** 2).mean(axis=1)).sort_values()
        rows.append({"test_sample": sid, **{f"nn{i + 1}": f"{d.index[i]} ({d.iloc[i]:.2f})" for i in range(min(k, len(d)))}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver

def _save(fig: Figure, path: Path) -> str:
    fig.savefig(path, dpi=110)
    return path.name


def run_eda(cfg: dict, out_dir: str | Path, with_features: bool = True, n_jobs: int | None = None) -> dict:
    """Write every EDA figure/table to `out_dir` and return the summary dict."""
    from phygrainnet.training.pipelines import Competition, feature_cache_path, image_feature_table

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comp = Competition(cfg)
    cat, labels, y = comp.catalog, comp.labels, comp.y
    ids = comp.train_ids
    groups = comp.groups
    ltab = label_table(labels, groups)
    ltab.to_csv(out / "label_table.csv", index=False)
    preds = loo_baselines(y, groups)
    figs = [_save(fig_label_curves(labels, ltab), out / "01_label_curves.png"),
            _save(fig_baseline_error(y, ids, preds), out / "02_baseline_error.png")]

    counts = photo_counts(cat)
    counts.to_csv(out / "photo_counts.csv", index=False)
    lt = ltab.set_index("sample_id")
    cams = cat.groupby("sample_id")["camera"].first()
    order = lt.sort_values("D50", na_position="first").index.tolist()
    titles = {s: f"{s} · D50 {lt.D50[s]:.3g} mm\n{cams[s]} · grav {lt.gravel[s] + lt.cobbles[s]:.0f}%" for s in order}
    figs.append(_save(fig_thumbnails(cat, order, titles, cfg, title="training soils sorted by D50 — 40 × 40 mm each"),
                      out / "03_train_thumbnails.png"))
    test_ids = comp.test_ids
    t_titles = {s: f"{s[:26]}\n{cams.get(s, '?')} · {int((cat.sample_id == s).sum())} photos" for s in test_ids}
    figs.append(_save(fig_thumbnails(cat, test_ids, t_titles, cfg, title="test soils (iPhone) — same 40 × 40 mm field"),
                      out / "04_test_thumbnails.png"))

    ct = colour_table(cat, float(cfg["data"].get("border_frac", 0.06)))
    ct.to_csv(out / "colour_table.csv", index=False)
    figs.append(_save(fig_colour_shift(ct), out / "05_colour_shift.png"))
    cam_stats = ct.groupby(["split", "camera"])[["raw_L", "raw_a", "raw_b", "gw_a", "gw_b", "sharpness"]].median().round(2)

    summary: dict = {
        "n_train": len(ids), "n_test": len(test_ids),
        "photos": counts.to_dict("records"),
        "cv_groups": int(len(np.unique(groups))),
        "families": ltab.family.value_counts().to_dict(),
        "fractions_mean": ltab[["clay", "silt", "sand", "gravel", "cobbles"]].mean().round(2).to_dict(),
        "D50_range_mm": [float(np.nanmin(ltab.D50)), float(np.nanmax(ltab.D50))],
        "baseline_logo_emd": {k: float(emd_per_sample(y, p).mean()) for k, p in preds.items()},
        "baseline_per_point_mean": dict(zip(TARGET_COLUMNS[:-1], per_point_contribution(y, preds["mean"]).round(3).tolist())),
        "camera_colour_medians": {f"{s}|{c}": v for (s, c), v in cam_stats.to_dict("index").items()},
    }

    if with_features:
        d = cfg["data"]
        feats = image_feature_table(comp, feature_cache_path(cfg), int(n_jobs or d.get("n_jobs", 2)))
        cols = feature_columns(feats)
        shift = feature_shift(feats, cols)
        corr = feature_label_corr(feats, cols, ltab)
        nn = nearest_training(feats, cols)
        shift.to_csv(out / "feature_shift.csv", index=False)
        corr.to_csv(out / "feature_label_corr.csv")
        nn.to_csv(out / "test_nearest_train.csv", index=False)
        figs.append(_save(fig_feature_space(feats, cols, shift, corr), out / "06_feature_space.png"))
        summary.update({
            "n_features": len(cols),
            "top_label_features": corr.head(10)[["log10_D50", "fines", "gravel"]].round(3).to_dict("index"),
            "top_shifted_features": shift.head(10).round(3).to_dict("records"),
            "share_features_shift_gt_1sd": float((shift.test_shift_sd.abs() > 1).mean()),
            "test_nearest_train": nn.to_dict("records"),
        })
    summary["figures"] = figs
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "eda_report.md").write_text(report_markdown(summary, ltab, counts), encoding="utf-8")
    return summary


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def report_markdown(s: dict, ltab: pd.DataFrame, counts: pd.DataFrame) -> str:
    lines = ["# EDA summary", "",
             f"- {s['n_train']} training soils in {s['cv_groups']} CV site groups; {s['n_test']} test soils.",
             f"- Families: {s['families']}. Mean fractions (%): {s['fractions_mean']}.",
             f"- D50 range: {s['D50_range_mm'][0]:.3g} – {s['D50_range_mm'][1]:.3g} mm.",
             f"- Image-free LOGO baselines: " + ", ".join(f"{k} {v:.2f}" for k, v in s["baseline_logo_emd"].items()) + ".",
             "", "## Photos", "", _table(counts)]
    if "n_features" in s:
        lines += ["", "## Features", "",
                  f"- {s['n_features']} usable photo features; {s['share_features_shift_gt_1sd']:.0%} shift by more than 1 train-SD on the test photos.",
                  "- Strongest label correlates: " + ", ".join(list(s["top_label_features"])[:5]) + ".",
                  "- Largest train→test shifts: " + ", ".join(r["feature"] for r in s["top_shifted_features"][:5]) + "."]
    lines += ["", "Figures: " + ", ".join(s["figures"]), ""]
    return "\n".join(lines)
