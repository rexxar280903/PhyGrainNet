"""End-to-end pipelines used by the scripts/ entry points."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from phygrainnet.config import get
from phygrainnet.data.catalog import build_catalog, catalog_problems
from phygrainnet.data.imaging import canonical_image_cached
from phygrainnet.data.labels import label_matrix, load_test_ids, load_train_labels, make_folds
from phygrainnet.experiment import append_registry, registry_row, write_outputs
from phygrainnet.features.granulometry import image_features
from phygrainnet.models.classical import build_classical
from phygrainnet.postprocess import make_valid, pointwise_median


class Competition:
    """Everything about the competition data that every pipeline needs."""

    def __init__(self, cfg: dict) -> None:
        d = cfg["data"]
        self.cfg = cfg
        self.root = Path(d["root"])
        self.catalog = build_catalog(self.root, scale_mode=d.get("scale_mode", "resize"))
        self.labels = load_train_labels(self.root)
        self.test_ids = load_test_ids(self.root)
        self.train_ids = self.labels["sample_id"].tolist()
        problems = catalog_problems(self.catalog, self.train_ids, self.test_ids)
        for p in problems:
            print(f"[catalog] WARNING: {p}")
        if any("without " in p or "no photos" in p for p in problems) and d.get("strict_catalog", True):
            raise RuntimeError("catalog problems: " + "; ".join(problems))
        if "ppm" not in self.catalog or not np.isfinite(self.catalog.ppm).all() or (self.catalog.ppm <= 0).any():
            raise ValueError("invalid physical scale; run the dataset audit")
        self.y = label_matrix(self.labels)
        if (not np.isfinite(self.y).all() or np.any((self.y < 0) | (self.y > 100))
                or np.any(np.diff(self.y, axis=1) < 0) or np.any(self.y[:, -1] != 100)):
            raise ValueError("invalid training labels; run the dataset audit")
        if len(set(self.train_ids)) != len(self.train_ids) or len(set(self.test_ids)) != len(self.test_ids):
            raise ValueError("duplicate sample IDs")
        if set(self.train_ids) & set(self.test_ids):
            raise ValueError("train/test sample overlap")
        cv = cfg.get("cv", {})
        self.groups, self.folds = make_folds(
            self.train_ids, cv.get("strategy", "logo"), int(cv.get("n_folds", 5)), int(cfg.get("seed", 42)),
            int(cv.get("max_gap", 10)),
        )
        for trn, val in self.folds:
            if not len(trn) or not len(val) or set(self.groups[trn]) & set(self.groups[val]):
                raise ValueError("invalid grouped CV fold")

    def fold_splits(self) -> list[dict]:
        return [{"fold": k, "train_ids": [self.train_ids[i] for i in trn],
                 "val_ids": [self.train_ids[i] for i in val]} for k, (trn, val) in enumerate(self.folds)]

    def canonical(self, row) -> np.ndarray:
        d = self.cfg["data"]
        return canonical_image_cached(row.path, float(row.ppm), float(d["target_ppm"]), float(d.get("border_frac", 0.06)),
                                      d.get("color_mode", "grayworld"), d.get("cache_dir"))


# ----------------------------------------------------------------------------- features

def _features_for_row(path: str, ppm: float, d: dict) -> dict:
    img = canonical_image_cached(path, ppm, float(d["target_ppm"]), float(d.get("border_frac", 0.06)),
                                 d.get("color_mode", "grayworld"), d.get("cache_dir"))
    return image_features(img, float(d["target_ppm"]))


def image_feature_table(comp: Competition, cache_csv: str | Path | None = None, n_jobs: int = 2) -> pd.DataFrame:
    if cache_csv and Path(cache_csv).exists():
        cached = pd.read_csv(cache_csv)
        if set(cached["path"]) >= set(comp.catalog["path"]):
            return cached
    d = comp.cfg["data"]
    rows = list(comp.catalog.itertuples())
    feats = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_features_for_row)(r.path, float(r.ppm), d) for r in rows)
    df = pd.DataFrame(feats)
    meta = comp.catalog[["path", "file", "split", "sample_id", "camera"]].reset_index(drop=True)
    out = pd.concat([meta, df], axis=1)
    if cache_csv:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_csv, index=False)
    return out


def sample_features(img_feats: pd.DataFrame, ids: list[str]) -> np.ndarray:
    num = img_feats.drop(columns=["path", "file", "split", "camera"]).groupby("sample_id").mean()
    return num.loc[ids].to_numpy(dtype=np.float64)


def feature_names(img_feats: pd.DataFrame) -> list[str]:
    return [c for c in img_feats.columns if c not in {"path", "file", "split", "sample_id", "camera"}]


# ----------------------------------------------------------------------------- classical CV

def run_classical(cfg: dict, config_path: str = "") -> dict[str, dict]:
    t0 = time.time()
    comp = Competition(cfg)
    out_dir = Path(get(cfg, "output.dir", "/kaggle/working/phygrainnet/outputs"))
    d = cfg["data"]
    tag = f"ppm{d['target_ppm']}_{d.get('color_mode', 'grayworld')}_b{d.get('border_frac', 0.06)}_{d.get('scale_mode', 'resize')}"
    feats = image_feature_table(comp, out_dir / "features" / f"image_features_{tag}.csv", int(d.get("n_jobs", 2)))
    X = sample_features(feats, comp.train_ids)
    Xt = sample_features(feats, comp.test_ids)
    keep = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 1e-9)
    X, Xt = X[:, keep], Xt[:, keep]
    results = {}
    for spec in cfg["classical"]["models"]:
        name = spec["name"]
        oof = np.zeros_like(comp.y)
        for trn, val in comp.folds:
            m = build_classical(spec).fit(X[trn], comp.y[trn])
            oof[val] = m.predict(X[val])
        test_pred = build_classical(spec).fit(X, comp.y).predict(Xt)
        exp_id = f"{cfg.get('experiment_id', 'A')}_{name}"
        summary = write_outputs(out_dir, exp_id, comp.train_ids, comp.y, make_valid(oof), comp.groups,
                                comp.test_ids, make_valid(test_pred), cfg,
                                {"n_features": int(X.shape[1]), "fold_splits": comp.fold_splits()})
        append_registry(get(cfg, "output.registry", out_dir / "registry.csv"),
                        registry_row(exp_id, spec["kind"], config_path, int(cfg.get("seed", 42)), summary,
                                     (time.time() - t0) / 60, notes=f"features={X.shape[1]}"))
        results[exp_id] = summary
        print(f"[classical] {exp_id}: CV EMD {summary['cv_emd']:.3f} (group std {summary['cv_emd_std_over_groups']:.2f})")
    return results


# ----------------------------------------------------------------------------- neural CV

def run_cnn(cfg: dict, config_path: str = "") -> dict:
    import torch

    from phygrainnet.training.data import records_from_catalog
    from phygrainnet.training.trainer import device_auto, predict_records, save_checkpoint, train_model

    t0 = time.time()
    comp = Competition(cfg)
    d = cfg["data"]
    out_dir = Path(get(cfg, "output.dir", "/kaggle/working/phygrainnet/outputs"))
    exp_id = cfg.get("experiment_id", "C000")
    device = device_auto()
    rec_args = dict(target_ppm=float(d["target_ppm"]), border_frac=float(d.get("border_frac", 0.06)),
                    color_mode=d.get("color_mode", "grayworld"), cache_dir=d.get("cache_dir"))
    train_recs = records_from_catalog(comp.catalog[comp.catalog.split == "train"], comp.labels,
                                      sample_ids=comp.train_ids, **rec_args)
    test_recs = records_from_catalog(comp.catalog[comp.catalog.split == "test"], None,
                                     sample_ids=comp.test_ids, **rec_args)
    extra_recs = []
    ext = cfg.get("external", {})
    if ext.get("use_ets_in_finetune") and ext.get("ets_catalog"):
        extra_recs = load_ets_records(cfg)

    init = get(cfg, "training.init_checkpoint")
    enc_only = bool(get(cfg, "training.init_encoder_only", False))
    oof = np.zeros_like(comp.y)
    fold_test = []
    histories = {}
    fold_limit = get(cfg, "cv.max_folds")
    if fold_limit is not None and int(fold_limit) < 1:
        raise ValueError("cv.max_folds must be at least 1")
    for k, (trn, val) in enumerate(comp.folds):
        if fold_limit is not None and k >= int(fold_limit):
            break
        model, hist = train_model(cfg, [train_recs[i] for i in trn] + extra_recs, [train_recs[i] for i in val],
                                  device, init, enc_only, log_prefix=f"[{exp_id} fold {k}] ")
        oof[val] = predict_records(cfg, model, [train_recs[i] for i in val], device)
        fold_test.append(predict_records(cfg, model, test_recs, device))
        histories[k] = hist
        if get(cfg, "output.save_checkpoints", True):
            save_checkpoint(out_dir / exp_id / f"fold{k}.pt", model, cfg)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if get(cfg, "training.full_fit", False):
        model, hist = train_model(cfg, train_recs + extra_recs, None, device, init, enc_only, log_prefix=f"[{exp_id} full] ")
        fold_test.append(predict_records(cfg, model, test_recs, device))
        save_checkpoint(out_dir / exp_id / "full.pt", model, cfg)
        histories["full"] = hist

    test_pred = pointwise_median(fold_test)
    done = [i for k, (_, val) in enumerate(comp.folds) if fold_limit is None or k < int(fold_limit) for i in val]
    done = np.array(sorted(done))
    summary = write_outputs(out_dir, exp_id, [comp.train_ids[i] for i in done], comp.y[done], make_valid(oof[done]),
                            comp.groups[done], comp.test_ids, make_valid(test_pred), cfg,
                            {"history": histories, "cv_complete": len(done) == len(comp.train_ids),
                             "n_train": len(comp.train_ids), "fold_splits": comp.fold_splits(),
                             "completed_folds": [k for k in histories if isinstance(k, int)]})
    from phygrainnet.models.phygrainnet import build_model

    n_params = sum(p.numel() for p in build_model(cfg).parameters())
    append_registry(get(cfg, "output.registry", out_dir / "registry.csv"),
                    registry_row(exp_id, cfg["model"]["name"], config_path, int(cfg.get("seed", 42)), summary,
                                 (time.time() - t0) / 60, n_params, notes=cfg.get("notes_short", "")))
    print(f"[cnn] {exp_id}: CV EMD {summary['cv_emd']:.3f} on {len(done)} samples")
    return summary


# ----------------------------------------------------------------------------- external data

def load_ets_records(cfg: dict):
    """ETS images are pre-resampled by scripts/prepare_ets.py (ppm already = target_ppm)."""
    from phygrainnet.training.data import records_from_catalog

    ext = cfg["external"]
    cat = pd.read_csv(ext["ets_catalog"])
    base = Path(ext["ets_catalog"]).parent
    cat["path"] = [p if Path(p).is_absolute() else str(base / p) for p in cat["path"]]
    lab = pd.read_csv(ext["ets_labels"])
    lab["sample_id"] = lab["sample_id"].astype(str)
    cat["sample_id"] = cat["sample_id"].astype(str)
    if ext.get("moisture"):
        cat = cat[cat["moisture"].isin(ext["moisture"] if isinstance(ext["moisture"], list) else [ext["moisture"]])]
    per = int(ext.get("max_images_per_sample", 6))
    cat = cat.groupby("sample_id").head(per)
    ids = sorted(set(cat["sample_id"]) & set(lab["sample_id"]))
    if ext.get("max_samples"):
        ids = ids[: int(ext["max_samples"])]
    d = cfg["data"]
    return records_from_catalog(cat, lab, float(d["target_ppm"]), 0.0, d.get("color_mode", "grayworld"),
                                d.get("cache_dir"), sample_ids=ids, source="ets")


def run_ets_pretrain(cfg: dict) -> str:
    """Supervised pretraining on ETS (masked loss on the 6-7 points it covers)."""
    from phygrainnet.training.trainer import device_auto, save_checkpoint, train_model

    recs = load_ets_records(cfg)
    n_val = max(1, int(len(recs) * 0.1))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    order = rng.permutation(len(recs))
    val = [recs[i] for i in order[:n_val]]
    trn = [recs[i] for i in order[n_val:]]
    pre_cfg = dict(cfg)
    pre_cfg["training"] = {**cfg["training"], **cfg.get("ets_pretrain", {})}
    model, hist = train_model(pre_cfg, trn, val, device_auto(), get(cfg, "training.init_checkpoint"),
                              bool(get(cfg, "training.init_encoder_only", True)), log_prefix="[ets] ")
    out = Path(get(cfg, "output.dir", "/kaggle/working/phygrainnet/outputs")) / cfg.get("experiment_id", "P100") / "ets_pretrained.pt"
    pre_cfg["pretrain_data"] = list(cfg.get("pretrain_data", [])) + ["ETS photogranulometry (CC BY 4.0)"]
    save_checkpoint(out, model, pre_cfg, {"history": hist})
    print(f"[ets] saved {out}")
    return str(out)


def run_ssl(cfg: dict) -> str:
    from phygrainnet.training.ssl import pretrain_simclr
    from phygrainnet.training.trainer import device_auto, save_checkpoint

    images = []
    s = cfg["ssl"]
    if s.get("use_competition_photos", True):
        comp = Competition(cfg)
        splits = ["train", "test"] if s.get("include_test_photos", True) else ["train"]
        for r in comp.catalog[comp.catalog.split.isin(splits)].itertuples():
            images.append(comp.canonical(r))
    if s.get("use_ets", False):
        for rec in load_ets_records(cfg):
            images.extend(rec.images)
    print(f"[ssl] {len(images)} images")
    model, hist = pretrain_simclr(cfg, images, device_auto())
    out = Path(get(cfg, "output.dir", "/kaggle/working/phygrainnet/outputs")) / cfg.get("experiment_id", "P000") / "ssl_encoder.pt"
    data = ["competition photos (unlabelled)"] + (["ETS photogranulometry (CC BY 4.0)"] if s.get("use_ets") else [])
    save_checkpoint(out, model, {**cfg, "pretrain_data": data}, {"history": hist, "encoders_only": True})
    print(f"[ssl] saved {out}")
    return str(out)
