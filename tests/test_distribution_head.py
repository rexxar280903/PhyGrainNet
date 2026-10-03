import torch

from phygrainnet.models.distribution_head import ConstrainedDistributionHead


def test_constrained_head_is_monotonic_and_ends_at_100():
    torch.manual_seed(42)
    head = ConstrainedDistributionHead(in_features=16, num_bins=11)
    x = torch.randn(8, 16)
    out = head(x)
    cumulative = out["cumulative"]
    masses = out["masses"]

    assert cumulative.shape == (8, 11)
    assert torch.all(masses >= 0)
    assert torch.all(cumulative[:, 1:] >= cumulative[:, :-1])
    assert torch.allclose(cumulative[:, -1], torch.full((8,), 100.0), atol=1e-4)
