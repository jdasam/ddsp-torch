import torch
from ddsp.losses import multiscale_fft, spectral_loss


def test_multiscale_fft_length():
    signal = torch.randn(2, 16000)
    stfts = multiscale_fft(signal, scales=[256, 512, 1024], overlap=0.75)
    assert len(stfts) == 3


def test_multiscale_fft_freq_bins():
    signal = torch.randn(2, 16000)
    stfts = multiscale_fft(signal, scales=[512], overlap=0.75)
    assert stfts[0].shape[1] == 512 // 2 + 1   # n_freq bins


def test_spectral_loss_identical_is_near_zero():
    signal = torch.randn(2, 16000)
    loss = spectral_loss(signal, signal, scales=[256, 512], overlap=0.75)
    assert loss.item() < 1e-3


def test_spectral_loss_different_is_positive():
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)
    loss = spectral_loss(x, y, scales=[256, 512], overlap=0.75)
    assert loss.item() > 0.1
