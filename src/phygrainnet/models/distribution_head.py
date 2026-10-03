from __future__ import annotations

import torch
from torch import nn


class ConstrainedDistributionHead(nn.Module):
    """Predict non-negative interval masses and convert them to a valid cumulative GSD."""

    def __init__(self, in_features: int, num_bins: int = 11) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, num_bins)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.projection(x)
        masses = torch.softmax(logits, dim=-1) * 100.0
        cumulative = torch.cumsum(masses, dim=-1)
        return {"logits": logits, "masses": masses, "cumulative": cumulative}
