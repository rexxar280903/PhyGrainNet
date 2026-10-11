"""Auditable compute counts and synthetic timings, never a dataset benchmark."""
from __future__ import annotations

import platform
import statistics
import time

import torch
from torch import nn

from phygrainnet.models.phygrainnet import build_model


def hardware_info() -> dict:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "vram_gib": torch.cuda.get_device_properties(i).total_memory / 2**30,
                  "capability": list(torch.cuda.get_device_capability(i))}
                 for i in range(torch.cuda.device_count())],
        "execution": "single GPU (cuda:0) or CPU; no DDP; GPU memory is not pooled",
    }


def model_profile(cfg: dict, device: str = "cpu") -> dict:
    """Count Conv2d/Linear MACs by observed shapes, excluding elementwise ops.

    MAC is one multiplication followed by an addition. Approx FLOPs = 2*MACs.
    Functional Sobel/Laplacian, normalization, activation, pooling and backward
    are excluded; this is not a full hardware performance model.
    """
    model = build_model(cfg).to(device).eval()
    t, d = cfg["training"], cfg["data"]
    b, tiles, scales, px = int(t["batch_size"]), int(t["tiles_per_sample"]), len(d["fields_mm"]), int(d["tile_px"])
    shape = [b, tiles, scales, 3, px, px]
    counts, handles = [], []

    def hook(name):
        def collect(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                macs = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            else:
                macs = output.numel() * module.in_features
            counts.append({"layer": name, "type": type(module).__name__, "shape": list(output.shape), "macs": int(macs)})
        return collect

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        x = torch.zeros(shape, device=device)
        with torch.no_grad():
            output = model(x if cfg["model"]["name"] == "phygrainnet_mv" else x[:, 0, 0])
    finally:
        for handle in handles:
            handle.remove()
    parameters = sum(p.numel() for p in model.parameters())
    macs = sum(row["macs"] for row in counts)
    return {"kind": "analytical counts observed on synthetic tensors", "input_shape": shape,
            "parameters": parameters, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "weight_mib_fp32": parameters * 4 / 2**20,
            "adamw_plus_ema_state_mib_fp32": parameters * 20 / 2**20,
            "state_memory_note": "5 FP32 arrays: weights, gradients, Adam m/v, EMA; excludes activations, buffers, allocator, AMP temporaries",
            "input_mib_fp32": x.numel() * 4 / 2**20, "forward_macs_batch": macs,
            "forward_flops_approx_batch": 2 * macs, "forward_macs_sample": macs // b,
            "output_shape": list(output["cumulative"].shape), "layers": counts,
            "count_excludes": "functional texture filters, normalization, activations, pooling, backward, optimizer, IO"}


def benchmark(cfg: dict, device: str = "auto", warmup: int = 2, steps: int = 5) -> dict:
    """Time synthetic AdamW training steps including EMA, with CUDA sync."""
    if warmup < 0 or steps < 1:
        raise ValueError("warmup >= 0 and steps >= 1 required")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: enable a compatible Kaggle GPU")
    from phygrainnet.losses import masked_emd_loss
    from phygrainnet.training.trainer import EMA

    model = build_model(cfg).to(device).train()
    t, d = cfg["training"], cfg["data"]
    b = int(t["batch_size"])
    x = torch.randn(b, int(t["tiles_per_sample"]), len(d["fields_mm"]), 3, int(d["tile_px"]), int(d["tile_px"]), device=device)
    if cfg["model"]["name"] != "phygrainnet_mv":
        x = x[:, 0, 0]
    target = torch.linspace(0, 100, 11, device=device).repeat(b, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]), weight_decay=float(t["weight_decay"]))
    amp = device.startswith("cuda") and bool(t.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, float(t.get("ema_decay", 0.995))) if t.get("ema_decay", 0) > 0 else None
    timings = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            output = model(x)["cumulative"]
        loss = masked_emd_loss(output.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("synthetic benchmark has non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("gradient_clip_norm", 1)))
        scaler.step(opt)
        scaler.update()
        if ema:
            ema.update(model)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= warmup:
            timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    return {"kind": "measured synthetic training; not real-data training", "device": device, "amp": amp,
            "warmup_steps": warmup, "measured_steps": steps, "seconds_per_step": median,
            "min_seconds_per_step": min(timings), "max_seconds_per_step": max(timings),
            "estimated_training_hours_per_fold": median * int(t["epochs"]) * int(t["steps_per_epoch"]) / 3600,
            "estimate_excludes": "image decode, canonical cache, augmentations, data loading, validation, test inference, checkpoint IO",
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.startswith("cuda") else None,
            "hardware": hardware_info()}
