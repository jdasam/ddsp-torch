import torch
import pytest
from ddsp.model import DDSP


@pytest.fixture
def small():
    return DDSP(hidden_size=64, n_harmonics=10, n_noise_bands=8,
                sampling_rate=16000, block_size=160)


def test_forward_shape(small):
    pitch = torch.rand(2, 100, 1) * 400 + 100
    loudness = torch.randn(2, 100, 1)
    out = small(pitch, loudness)
    assert out.shape == (2, 100 * 160, 1)


def test_output_finite(small):
    pitch = torch.full((1, 50, 1), 220.0)
    loudness = torch.zeros(1, 50, 1)
    out = small(pitch, loudness)
    assert torch.isfinite(out).all()


def test_gradients_flow(small):
    pitch = torch.rand(1, 50, 1) * 300 + 100
    loudness = torch.zeros(1, 50, 1)
    out = small(pitch, loudness)
    out.pow(2).mean().backward()
    assert any(p.grad is not None for p in small.parameters())


def test_reverb_shape(small):
    x = torch.randn(2, 8000, 1)
    assert small.reverb(x).shape == x.shape
