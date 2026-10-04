"""Training / inference loops for PhyGrainNet (random init, no external weights)."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phygrainnet.losses import masked_emd_loss
from phygrainnet.models.phygrainnet import PhyGrainNetMV, build_model
from phygrainnet.training.data import SampleRecord, TrainTileDataset, inference_tiles

CHECKPOINT_ORIGIN = "phygrainnet-from-scratch"


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fields_px(cfg: dict) -> list[int]:
    ppm = float(cfg["data"]["target_ppm"])
    return [int(round(f * ppm)) for f in cfg["data"]["fields_mm"]]


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "origin": CHECKPOINT_ORIGIN,
            "pretrain_data": cfg.get("pretrain_data", []),
            "config": cfg,
            "state_dict": model.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("origin") != CHECKPOINT_ORIGIN:
        # DEC-001 guard: refuse weights that were not produced by this repository.
        raise ValueError(f"{path} is not a {CHECKPOINT_ORIGIN} checkpoint (origin={ckpt.get('origin')!r})")
    return ckpt


def init_from(model: torch.nn.Module, path: str | Path | None, encoder_only: bool) -> None:
    if not path:
        return
    ckpt = load_checkpoint(path)
    state = ckpt["state_dict"]
    if encoder_only or "encoders_only" in ckpt:
        enc = {k[len("encoders."):]: v for k, v in state.items() if k.startswith("encoders.")}
        model.encoders.load_state_dict(enc, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[init] ignored unexpected keys: {unexpected[:5]}")


def _lr_lambda(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


def train_model(
    cfg: dict,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord] | None = None,
    device: torch.device | None = None,
    init_checkpoint: str | None = None,
    encoder_only_init: bool = False,
    log_prefix: str = "",
) -> tuple[torch.nn.Module, list[dict]]:
    device = device or device_auto()
    t = cfg["training"]
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    init_from(model, init_checkpoint, encoder_only_init)

    ds = TrainTileDataset(
        train_records,
        fields_px(cfg),
        int(cfg["data"]["tile_px"]),
        int(t["tiles_per_sample"]),
        int(t["steps_per_epoch"]) * int(t["batch_size"]),
        seed=seed,
        scale_jitter=float(t.get("scale_jitter", 0.08)),
        photometric=float(t.get("photometric", 1.0)),
        mix_prob=float(t.get("mix_prob", 0.3)),
    )
    dl = DataLoader(
        ds,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 2)),
        drop_last=True,
        persistent_workers=int(cfg["data"].get("num_workers", 2)) > 0,
    )
    epochs = int(t["epochs"])
    total = epochs * len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(t.get("warmup_steps", 100))))
    use_amp = device.type == "cuda" and bool(t.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, float(t.get("ema_decay", 0.995))) if float(t.get("ema_decay", 0)) > 0 else None

    history = []
    eval_every = int(t.get("eval_every", max(1, epochs // 5)))
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        t0, losses = time.time(), []
        for batch in dl:
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = masked_emd_loss(
                out["cumulative"].float(),
                batch["target"].to(device),
                batch["mask"].to(device),
                batch["upper_bound"].to(device),
                batch["finest_index"].to(device),
                bound_weight=float(t.get("bound_weight", 1.0)),
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("gradient_clip_norm", 1.0)))
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.detach()))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "sec": time.time() - t0}
        if val_records and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            from phygrainnet.metrics.kaggle_emd import emd_per_sample

            m = ema.shadow if ema is not None else model
            pred = predict_records(cfg, m, val_records, device)
            y = np.stack([r.target for r in val_records])
            mk = np.stack([r.mask for r in val_records])
            rec["val_emd"] = float(np.mean(emd_per_sample(np.where(mk > 0, y, pred), pred)))
        history.append(rec)
        print(f"{log_prefix}epoch {epoch + 1}/{epochs} " + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k != "epoch"), flush=True)

    return (ema.shadow if ema is not None else model), history


@torch.no_grad()
def predict_records(
    cfg: dict,
    model: torch.nn.Module,
    records: list[SampleRecord],
    device: torch.device | None = None,
    tta: int | None = None,
) -> np.ndarray:
    """Average cumulative prediction over all grid tiles and `tta` rotations/flips."""
    device = device or device_auto()
    model.eval()
    inf = cfg.get("inference", {})
    tta = int(inf.get("tta", 4) if tta is None else tta)
    chunk = int(inf.get("chunk", 64))
    preds = []
    for rec in records:
        tiles = inference_tiles(rec, fields_px(cfg), int(cfg["data"]["tile_px"]), int(inf.get("max_tiles_per_image", 48)))
        outs = []
        for k in range(max(1, tta)):
            tt = torch.rot90(tiles, k % 4, dims=(-2, -1))
            if k >= 4:
                tt = torch.flip(tt, dims=(-1,))
            if isinstance(model, PhyGrainNetMV):
                embs = []
                for i in range(0, tt.shape[0], chunk):
                    xb = tt[i : i + chunk].unsqueeze(0).to(device)
                    embs.append(model.encode_tiles(xb)[0])
                e = torch.cat(embs).unsqueeze(0)
                z = model.mlp(model.pool(e))
                outs.append(model.head(z)["cumulative"][0].float().cpu().numpy())
            else:
                xb = tt[:, 0].to(device)
                outs.append(model(xb)["cumulative"].float().mean(0).cpu().numpy())
        preds.append(np.mean(outs, axis=0))
    return np.stack(preds)
