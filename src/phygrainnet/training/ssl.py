"""Self-supervised pretraining of the GrainEncoder from random initialisation (SimCLR).

Uses only soil photographs (competition train + test photos without labels, and
external soil datasets), so the encoder never sees generic-image weights (DEC-001).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from phygrainnet.models.phygrainnet import GrainEncoder, PhyGrainNetMV
from phygrainnet.training.data import UnlabeledTileDataset


class SimCLR(nn.Module):
    def __init__(self, encoder: GrainEncoder, proj_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.encoder(x)), dim=1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain_simclr(
    cfg: dict,
    images: list[np.ndarray],
    device: torch.device,
) -> tuple[PhyGrainNetMV, list[dict]]:
    from phygrainnet.models.phygrainnet import build_model
    from phygrainnet.training.trainer import fields_px

    s = cfg["ssl"]
    torch.manual_seed(int(cfg.get("seed", 42)))
    full = build_model(cfg)
    encoder = full.encoders[0]
    model = SimCLR(encoder, int(s.get("proj_dim", 128))).to(device)
    ds = UnlabeledTileDataset(images, fields_px(cfg), int(s.get("tile_px", cfg["data"]["tile_px"])),
                              int(s["steps_per_epoch"]) * int(s["batch_size"]), seed=int(cfg.get("seed", 42)))
    nw = int(cfg["data"].get("num_workers", 2))
    dl = DataLoader(ds, batch_size=int(s["batch_size"]), num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    epochs = int(s["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(s.get("learning_rate", 1e-3)), weight_decay=float(s.get("weight_decay", 1e-4)))
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / max(1, total))))
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        t0, losses = time.time(), []
        for v1, v2 in dl:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z1, z2 = model(v1), model(v2)
            loss = nt_xent(z1.float(), z2.float(), float(s.get("temperature", 0.2)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        hist.append({"epoch": epoch, "ssl_loss": float(np.mean(losses)), "sec": time.time() - t0})
        print(f"[ssl] epoch {epoch + 1}/{epochs} loss={hist[-1]['ssl_loss']:.4f} ({hist[-1]['sec']:.0f}s)", flush=True)
    for k in range(1, len(full.encoders)):
        full.encoders[k].load_state_dict(encoder.state_dict())
    return full.cpu(), hist
