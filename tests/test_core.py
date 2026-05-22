import torch
import pytest
from ddsp.core import harmonic_synth, fft_convolve, filtered_noise, remove_above_nyquist, upsample


def test_remove_above_nyquist_kills_high_harmonics():
    # f0=4000Hz, harmonics 1-5 → 4k,8k,12k,16k,20k. Nyquist=8kHz → only H1 survives
    pitch = torch.full((1, 10, 1), 4000.0)
    amps = torch.ones(1, 10, 5)
    out = remove_above_nyquist(amps, pitch, sampling_rate=16000)
    assert out[0, 0, 0] > 0.5   # 4kHz survives
    assert out[0, 0, 1] < 0.01  # 8kHz killed


def test_harmonic_synth_shape():
    batch, n_samples, n_harm = 2, 16000, 10
    pitch = torch.full((batch, n_samples, 1), 220.0)
    amps = torch.rand(batch, n_samples, n_harm) * 0.1
    audio = harmonic_synth(pitch, amps, sampling_rate=16000)
    assert audio.shape == (batch, n_samples, 1)


def test_harmonic_synth_zero_amps_gives_silence():
    pitch = torch.full((1, 1600, 1), 440.0)
    amps = torch.zeros(1, 1600, 5)
    audio = harmonic_synth(pitch, amps, sampling_rate=16000)
    assert audio.abs().max().item() < 1e-6


def test_fft_convolve_shape():
    # Works on (..., n_samples) — test with 3D (batch, frames, block)
    x = torch.randn(2, 50, 160)
    k = torch.randn(2, 50, 160)
    out = fft_convolve(x, k)
    assert out.shape == (2, 50, 160)


def test_fft_convolve_2d():
    # Also works for (batch, n_samples) used by reverb
    x = torch.randn(2, 8000)
    k = torch.randn(1, 8000)
    out = fft_convolve(x, k)
    assert out.shape == (2, 8000)


def test_filtered_noise_shape():
    batch, n_frames, n_bands = 2, 100, 65
    mags = torch.rand(batch, n_frames, n_bands).abs() + 1e-4
    block_size = 160
    audio = filtered_noise(mags, block_size)
    assert audio.shape == (batch, n_frames * block_size, 1)


def test_upsample_shape():
    x = torch.randn(2, 100, 3)
    out = upsample(x, factor=160)
    assert out.shape == (2, 16000, 3)
