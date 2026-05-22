import math
import torch
import torch.nn as nn


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(x + eps)


def remove_above_nyquist(
    amplitudes: torch.Tensor,
    pitch: torch.Tensor,
    sampling_rate: int,
) -> torch.Tensor:
    """Zero amplitudes for harmonics above the Nyquist frequency.

    Args:
        amplitudes: (batch, frames, n_harmonics)
        pitch:      (batch, frames, 1) f0 in Hz
        sampling_rate: int

    Returns:
        (batch, frames, n_harmonics) — above-Nyquist harmonics zeroed
    """
    n_harm = amplitudes.shape[-1]
    idx = torch.arange(1, n_harm + 1, device=pitch.device, dtype=pitch.dtype)
    freqs = pitch * idx                              # (batch, frames, n_harm)
    mask = (freqs < sampling_rate / 2).float()
    return amplitudes * mask


def harmonic_synth(
    pitch: torch.Tensor,
    amplitudes: torch.Tensor,
    sampling_rate: int,
) -> torch.Tensor:
    """Additive harmonic synthesizer using cumulative phase integration.

    Args:
        pitch:      (batch, n_samples, 1) f0 in Hz at audio rate
        amplitudes: (batch, n_samples, n_harmonics) per-harmonic amplitudes at audio rate
        sampling_rate: int

    Returns:
        audio: (batch, n_samples, 1)
    """
    n_harm = amplitudes.shape[-1]
    idx = torch.arange(1, n_harm + 1, device=pitch.device, dtype=pitch.dtype)
    omega = 2 * math.pi * pitch * idx / sampling_rate   # phase increment per sample
    phase = torch.cumsum(omega, dim=1)                   # cumulative phase
    audio = (torch.sin(phase) * amplitudes).sum(dim=-1, keepdim=True)
    return audio


def fft_convolve(
    signal: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """FFT-based linear convolution along the last dimension.

    Handles any shape (..., n_samples). kernel must be broadcastable with signal.

    Args:
        signal: (..., n_samples)
        kernel: (..., n_samples)

    Returns:
        (..., n_samples)
    """
    n = signal.shape[-1]
    signal_padded = nn.functional.pad(signal, (0, n))
    kernel_padded = nn.functional.pad(kernel, (n, 0))   # left-pad for causal alignment
    out = torch.fft.irfft(torch.fft.rfft(signal_padded) * torch.fft.rfft(kernel_padded))
    return out[..., out.shape[-1] // 2: out.shape[-1] // 2 + n]


def amp_to_impulse_response(
    magnitudes: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Convert per-frame filter magnitudes to windowed impulse responses via IFFT.

    Args:
        magnitudes: (batch, n_frames, n_bands) positive values
        block_size: target IR length in samples

    Returns:
        ir: (batch, n_frames, block_size)
    """
    # Build Hermitian-symmetric spectrum and invert
    magnitudes = torch.cat([magnitudes, magnitudes[..., 1:-1].flip(-1)], dim=-1)
    ir = torch.fft.irfft(magnitudes, n=block_size)
    # Shift center to causal and apply Hann window
    ir = torch.roll(ir, ir.shape[-1] // 2, dims=-1)
    win = torch.hann_window(ir.shape[-1], device=ir.device, dtype=ir.dtype)
    return ir * win


def filtered_noise(
    magnitudes: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Filtered noise synthesizer: shape noise with per-frame spectral envelopes.

    Args:
        magnitudes: (batch, n_frames, n_bands) positive filter magnitudes
        block_size: samples per frame

    Returns:
        audio: (batch, n_frames * block_size, 1)
    """
    batch, n_frames, _ = magnitudes.shape
    ir = amp_to_impulse_response(magnitudes, block_size)          # (batch, n_frames, block_size)
    noise = torch.rand(batch, n_frames, block_size, device=magnitudes.device) * 2 - 1
    audio = fft_convolve(noise, ir)                               # (batch, n_frames, block_size)
    return audio.reshape(batch, n_frames * block_size, 1)


def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    """Linear interpolation upsample along time dimension.

    Args:
        signal: (batch, frames, channels)
        factor: integer upsampling ratio

    Returns:
        (batch, frames * factor, channels)
    """
    signal = signal.permute(0, 2, 1)                              # (batch, channels, frames)
    signal = nn.functional.interpolate(
        signal, size=signal.shape[-1] * factor, mode='linear', align_corners=False
    )
    return signal.permute(0, 2, 1)
