"""Training losses aligned with the official competition metric."""
from __future__ import annotations

import torch

# log10 widths of the 10 intervals between the 11 support points (official weights).
_DIAMETERS = torch.tensor([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0], dtype=torch.float64)
EMD_WEIGHTS = torch.diff(torch.log10(_DIAMETERS)).float()


def masked_emd_loss(
    pred_cdf: torch.Tensor,
    target_cdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    upper_bound: torch.Tensor | None = None,
    finest_index: torch.Tensor | None = None,
    bound_weight: float = 1.0,
) -> torch.Tensor:
    """Official log-weighted EMD on the points that have labels.

    pred_cdf, target_cdf : (B, 11) cumulative % passing; NaN targets are allowed
                           where mask == 0.
    mask                 : (B, 11) 1 = labelled point. None = all labelled.
    upper_bound          : (B,) % passing the finest external sieve (e.g. 80 µm).
                           Unlabelled finer points cannot exceed it.
    finest_index         : (B,) index of the first *labelled* support point; the
                           bound applies to points below it.
    """
    w = EMD_WEIGHTS.to(pred_cdf.device)
    p, t = pred_cdf[:, :-1], target_cdf[:, :-1]
    if mask is None:
        m = torch.ones_like(p)
    else:
        m = mask[:, :-1].to(p.dtype)
    t = torch.nan_to_num(t, nan=0.0)
    loss = (torch.abs(p - t) * m * w).sum(dim=1)

    if upper_bound is not None and finest_index is not None:
        idx = torch.arange(p.shape[1], device=p.device).unsqueeze(0)
        below = (idx < finest_index.unsqueeze(1)).to(p.dtype)
        excess = torch.relu(p - upper_bound.unsqueeze(1).to(p.dtype)) * below * w
        loss = loss + bound_weight * excess.sum(dim=1)
    return loss.mean()
