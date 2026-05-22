import math
import torch
import pytest
from ddsp.core import (
    harmonic_synth, fft_convolve, filtered_noise, remove_above_nyquist, upsample,
    frequency_impulse_response, apply_window_to_impulse_response, fft_convolve_ola,
    upsample_with_windows, angular_cumsum,
)


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


def test_fft_convolve_2d():
    # Also works for (batch, n_samples) used by reverb
    x = torch.randn(2, 8000)
    k = torch.randn(1, 8000)
    out = fft_convolve(x, k)
    assert out.shape == (2, 8000)


def test_upsample_shape():
    x = torch.randn(2, 100, 3)
    out = upsample(x, factor=160)
    assert out.shape == (2, 16000, 3)


def test_frequency_impulse_response_shape():
    # 65 one-sided bands → IR of 2*(65-1)=128 samples
    mags = torch.rand(2, 100, 65).abs() + 1e-4
    ir = frequency_impulse_response(mags)
    assert ir.shape == (2, 100, 128)


def test_frequency_impulse_response_finite():
    mags = torch.rand(2, 100, 65).abs() + 1e-4
    ir = frequency_impulse_response(mags)
    assert torch.isfinite(ir).all()


def test_fft_convolve_ola_shape():
    # signal: (batch, n_samples), ir: (batch, n_frames, ir_size)
    batch, n_samples, n_frames, ir_size = 2, 16000, 100, 128
    signal = torch.randn(batch, n_samples)
    ir = torch.randn(batch, n_frames, ir_size)
    out = fft_convolve_ola(signal, ir)
    assert out.shape == (batch, n_samples)


def test_fft_convolve_ola_finite():
    signal = torch.randn(2, 16000)
    ir = torch.randn(2, 100, 128)
    out = fft_convolve_ola(signal, ir)
    assert torch.isfinite(out).all()


def test_filtered_noise_shape():
    # Existing test — must still pass unchanged
    batch, n_frames, n_bands = 2, 100, 65
    mags = torch.rand(batch, n_frames, n_bands).abs() + 1e-4
    block_size = 160
    audio = filtered_noise(mags, block_size)
    assert audio.shape == (batch, n_frames * block_size, 1)


def test_upsample_with_windows_shape():
    # (batch=2, n_frames=100, channels=10) → (2, 16000, 10)
    x = torch.randn(2, 100, 10)
    out = upsample_with_windows(x, 100 * 160)
    assert out.shape == (2, 16000, 10)


def test_upsample_with_windows_smooth():
    # Step input: all-zero frames then all-one frames.
    # Window upsampling should produce a smooth ramp — no abrupt jumps
    x = torch.zeros(1, 20, 1)
    x[:, 10:, :] = 1.0
    out = upsample_with_windows(x, 20 * 160)
    # No sample should jump by more than 0.015 in one step
    diffs = (out[:, 1:, :] - out[:, :-1, :]).abs()
    assert diffs.max().item() < 0.015


def test_filtered_noise_lowpass():
    torch.manual_seed(0)
    mags = torch.zeros(1, 50, 65)
    mags[:, :, :5] = 1.0  # only low bands active
    audio = filtered_noise(mags, 160).squeeze(-1).squeeze(0)  # (8000,)
    spec = torch.fft.rfft(audio).abs()
    low_energy = spec[:10].mean()
    high_energy = spec[100:].mean()
    assert low_energy > 5 * high_energy, f"Expected low-pass attenuation: low={low_energy:.4f}, high={high_energy:.4f}"


def test_angular_cumsum_shape():
    omega = torch.rand(2, 6400, 10) * 0.01  # (batch, samples, harmonics)
    phase = angular_cumsum(omega)
    assert phase.shape == (2, 6400, 10)


def test_angular_cumsum_range():
    # All values must be in [0, 2π]
    omega = torch.rand(2, 3000, 5) * 0.1
    phase = angular_cumsum(omega)
    assert phase.min().item() >= 0.0
    assert phase.max().item() <= 2 * math.pi + 1e-5


def test_angular_cumsum_matches_cumsum_for_short_seq():
    # For sequences shorter than chunk_size, results should match plain cumsum mod 2π
    omega = torch.rand(1, 100, 3) * 0.01
    phase_simple = torch.cumsum(omega, dim=1) % (2 * math.pi)
    phase_angular = angular_cumsum(omega, chunk_size=1000)
    assert torch.allclose(phase_simple, phase_angular, atol=1e-5)


def test_angular_cumsum_reduces_drift_vs_plain():
    # angular_cumsum should reduce float32 phase error vs plain cumsum.
    # The within-chunk mod keeps magnitudes bounded, improving float32 precision.
    # Measured with circular (shortest-arc) distance to correctly handle phase wrap.
    sr, f0, n = 16000, 440.0, 100_000
    omega_f64 = torch.full((1, n, 1), 2 * math.pi * f0 / sr, dtype=torch.float64)
    omega_f32 = omega_f64.float()
    gt = torch.cumsum(omega_f64, dim=1) % (2 * math.pi)
    plain = torch.cumsum(omega_f32, dim=1) % (2 * math.pi)
    ac = angular_cumsum(omega_f32)

    def circular_mae(a, b):
        d = (a - b) % (2 * math.pi)
        d[d > math.pi] -= 2 * math.pi
        return d.abs().mean()

    err_plain = circular_mae(plain, gt.float())
    err_ac = circular_mae(ac, gt.float())
    improvement = err_plain / (err_ac + 1e-10)
    assert improvement > 2.5, (
        f"Expected ≥2.5x drift improvement: got {improvement:.2f}x "
        f"(ac={err_ac:.2e}, plain={err_plain:.2e})"
    )


def test_angular_cumsum_no_boundary_discontinuity():
    # Phase steps at chunk boundaries should equal ω (no discontinuity)
    chunk_size = 100
    n = chunk_size * 5  # exactly 5 chunks
    omega = torch.rand(1, n, 2) * 0.05 + 0.001  # (batch, samples, harmonics)
    phase = angular_cumsum(omega, chunk_size=chunk_size)
    # Step size at any sample = phase[t] - phase[t-1] mod 2π should equal omega[t]
    # Check specifically at chunk boundaries (samples chunk_size, 2*chunk_size, ...)
    for boundary in range(chunk_size, n, chunk_size):
        step = (phase[:, boundary, :] - phase[:, boundary - 1, :]) % (2 * math.pi)
        expected = omega[:, boundary, :] % (2 * math.pi)
        assert torch.allclose(step, expected, atol=1e-4), (
            f"Discontinuity at boundary {boundary}: step={step}, expected={expected}"
        )
