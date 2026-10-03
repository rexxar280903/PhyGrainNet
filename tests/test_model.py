import torch

from phygrainnet.models.phygrainnet import PhyGrainNetV0


def test_phygrainnet_v0_forward():
    model = PhyGrainNetV0()
    x = torch.randn(2, 3, 192, 192)
    out = model(x)
    assert out["cumulative"].shape == (2, 11)
    assert out["embedding"].shape == (2, 128)
