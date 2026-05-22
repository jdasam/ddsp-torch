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


def angular_cumsum(
    angular_frequency: torch.Tensor,
    chunk_size: int = 1000,
) -> torch.Tensor:
    """Chunk-based cumsum with mod 2π to prevent phase accumulation errors.

    Plain cumsum accumulates float32 errors that become audible for sequences
    longer than ~100k samples. This function splits the sequence into chunks,
    applies cumsum within each chunk, then adds mod-2π offsets between chunks.

    Matches Magenta's angular_cumsum (chunk_size=1000 default).

    Args:
        angular_frequency: (batch, n_samples, ...) radians per sample
        chunk_size: samples per chunk; smaller → less drift, slightly slower
    Returns:
        phase: same shape as input, values in [0, 2π]
    """
    shape = angular_frequency.shape
    n_time = shape[1]
    extra_dims = shape[2:]

    # Pad to multiple of chunk_size
    remainder = n_time % chunk_size
    if remainder:
        pad_len = chunk_size - remainder
        angular_frequency = nn.functional.pad(
            angular_frequency.reshape(shape[0], n_time, -1),
            (0, 0, 0, pad_len),
        ).reshape(shape[0], n_time + pad_len, *extra_dims)

    n_time_padded = angular_frequency.shape[1]
    n_chunks = n_time_padded // chunk_size

    # Reshape into chunks: (batch, n_chunks, chunk_size, ...)
    chunks = angular_frequency.reshape(shape[0], n_chunks, chunk_size, *extra_dims)
    phase = torch.cumsum(chunks, dim=2)

    # Offset for next chunk = last sample of current chunk, mod 2π
    # Shape: (batch, n_chunks, 1, ...)
    offsets = phase[:, :, -1:, ...] % (2 * math.pi)
    # Prepend zero offset for the first chunk
    zero = torch.zeros_like(offsets[:, :1, ...])
    offsets = torch.cat([zero, offsets[:, :-1, ...]], dim=1)
    # Cumulative offsets between chunks
    offsets = torch.cumsum(offsets, dim=1) % (2 * math.pi)

    phase = (phase + offsets) % (2 * math.pi)
    phase = phase.reshape(shape[0], n_time_padded, *extra_dims)

    if remainder:
        phase = phase[:, :n_time, ...]
    return phase


def harmonic_synth(
    pitch: torch.Tensor,
    amplitudes: torch.Tensor,
    sampling_rate: int,
    use_angular_cumsum: bool = False,
) -> torch.Tensor:
    """Additive harmonic synthesizer using cumulative phase integration.

    Args:
        pitch:      (batch, n_samples, 1) f0 in Hz at audio rate
        amplitudes: (batch, n_samples, n_harmonics) per-harmonic amplitudes at audio rate
        sampling_rate: int
        use_angular_cumsum: use chunk-based cumsum to avoid phase drift for
            long sequences (>100k samples). Recommended for inference.
    Returns:
        audio: (batch, n_samples, 1)
    """
    n_harm = amplitudes.shape[-1]
    idx = torch.arange(1, n_harm + 1, device=pitch.device, dtype=pitch.dtype)
    omega = 2 * math.pi * pitch * idx / sampling_rate   # phase increment per sample
    if use_angular_cumsum:
        phase = angular_cumsum(omega)
    else:
        phase = torch.cumsum(omega, dim=1)               # cumulative phase
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


def apply_window_to_impulse_response(
    ir: torch.Tensor,
    window_size: int = 0,
) -> torch.Tensor:
    """Apply Hann window to a zero-phase IR and convert to causal form.

    Matches Magenta's apply_window_to_impulse_response (causal=False path).

    Args:
        ir: (batch, n_frames, ir_size) — zero-phase IR from irfft
        window_size: Hann window length; 0 or > ir_size defaults to ir_size
    Returns:
        (batch, n_frames, ir_size) — causal windowed IR
    """
    ir_size = ir.shape[-1]
    if window_size <= 0 or window_size > ir_size:
        window_size = ir_size

    window = torch.hann_window(window_size, device=ir.device, dtype=ir.dtype)

    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([
            window[half_idx:],
            torch.zeros(padding, device=ir.device, dtype=ir.dtype),
            window[:half_idx],
        ])
    else:
        window = torch.roll(window, window_size // 2)  # zero-phase form

    ir = window * ir  # broadcasts over (batch, n_frames)

    ir = torch.roll(ir, ir_size // 2, dims=-1)  # shift to causal form (center peak at ir_size // 2)

    return ir


def frequency_impulse_response(
    magnitudes: torch.Tensor,
    window_size: int = 0,
) -> torch.Tensor:
    """Convert one-sided frequency magnitudes to windowed causal IRs.

    Matches Magenta's frequency_impulse_response.

    Args:
        magnitudes: (batch, n_frames, n_bands) — non-negative one-sided spectrum
        window_size: Hann window size (0 = full IR size = 2*(n_bands-1))
    Returns:
        (batch, n_frames, 2*(n_bands-1)) — causal windowed IR
    """
    magnitudes_c = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    ir = torch.fft.irfft(magnitudes_c)  # (batch, n_frames, 2*(n_bands-1))
    return apply_window_to_impulse_response(ir, window_size)


def fft_convolve_ola(
    signal: torch.Tensor,
    ir: torch.Tensor,
) -> torch.Tensor:
    """Frame-based FFT convolution with overlap-and-add (Magenta-style).

    Splits signal into non-overlapping frames, convolves each frame with its
    corresponding IR, then overlap-adds the results. Applies 'same' padding
    with group-delay compensation so output length == input length.

    Args:
        signal: (batch, n_samples)
        ir: (batch, n_frames, ir_size)
    Returns:
        (batch, n_samples)
    """
    batch, n_samples = signal.shape
    n_frames, ir_size = ir.shape[1], ir.shape[2]

    frame_size = int(math.ceil(n_samples / n_frames))

    # Pad signal so it divides evenly into n_frames
    padded_len = n_frames * frame_size
    signal_padded = nn.functional.pad(signal, (0, padded_len - n_samples))

    # Split into non-overlapping frames: (batch, n_frames, frame_size)
    audio_frames = signal_padded.reshape(batch, n_frames, frame_size)

    # Next power-of-2 FFT size covering convolved frame length
    fft_size = 2 ** int(math.ceil(math.log2(frame_size + ir_size - 1)))

    # Per-frame convolution in frequency domain
    audio_fft = torch.fft.rfft(audio_frames, n=fft_size)
    ir_fft = torch.fft.rfft(ir, n=fft_size)
    conv_frames = torch.fft.irfft(audio_fft * ir_fft, n=fft_size)  # (batch, n_frames, fft_size)

    # Overlap-and-add via fold
    # fold input: (N, kernel_size, L) → output: (N, 1, 1, out_len)
    ola_len = (n_frames - 1) * frame_size + fft_size
    conv_t = conv_frames.permute(0, 2, 1)  # (batch, fft_size, n_frames)
    audio_out = nn.functional.fold(
        conv_t,
        output_size=(1, ola_len),
        kernel_size=(1, fft_size),
        stride=(1, frame_size),
    ).squeeze(1).squeeze(1)  # (batch, ola_len)

    # Crop with group-delay compensation for linear-phase windowed FIR
    # Matches Magenta's crop_and_compensate_delay(padding='same', delay_compensation=-1)
    delay = (ir_size - 1) // 2 - 1
    crop = ola_len - n_samples
    end = crop - delay
    audio_out = audio_out[:, delay: ola_len - end]

    return audio_out


def filtered_noise(
    magnitudes: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Filtered noise synthesizer using time-varying FIR via OLA convolution.

    Matches Magenta's FilteredNoise.get_signal approach: generates a single
    noise signal and filters it with per-frame IRs using overlap-and-add,
    avoiding frame-boundary discontinuities.

    Args:
        magnitudes: (batch, n_frames, n_bands) — positive filter magnitudes
        block_size: samples per frame
    Returns:
        audio: (batch, n_frames * block_size, 1)
    """
    batch, n_frames, _ = magnitudes.shape
    n_samples = n_frames * block_size

    ir = frequency_impulse_response(magnitudes)  # (batch, n_frames, ir_size)
    noise = torch.rand(batch, n_samples, device=magnitudes.device, dtype=magnitudes.dtype) * 2 - 1
    audio = fft_convolve_ola(noise, ir)  # (batch, n_samples)
    return audio.unsqueeze(-1)


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


def upsample_with_windows(signal: torch.Tensor, n_timesteps: int) -> torch.Tensor:
    """Upsample using overlapping Hann windows (OLA). Smoother than linear for amplitude envelopes.

    Matches Magenta's upsample_with_windows. n_timesteps must be divisible by
    the number of input frames (always true when n_timesteps = n_frames * block_size).

    Args:
        signal: (batch, n_frames, channels)
        n_timesteps: target length; must satisfy n_timesteps % n_frames == 0
    Returns:
        (batch, n_timesteps, channels)
    """
    batch, n_frames, channels = signal.shape

    # Add endpoint (repeat last frame) so there are n_frames intervals
    signal = torch.cat([signal, signal[:, -1:, :]], dim=1)  # (batch, n_frames+1, channels)
    n_intervals = n_frames

    if n_timesteps % n_intervals != 0:
        raise ValueError(
            f'n_timesteps ({n_timesteps}) must be divisible by n_frames ({n_frames})'
        )

    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length, periodic=True, device=signal.device, dtype=signal.dtype)

    # Apply window to each frame: (batch, n_frames+1, channels, window_length)
    x_windowed = signal.unsqueeze(-1) * window  # broadcast (window_length,)

    # OLA into output buffer
    out_len = n_frames * hop_size + window_length
    output = signal.new_zeros(batch, out_len, channels)
    for i in range(n_frames + 1):
        start = i * hop_size
        # x_windowed[:, i, :, :] is (batch, channels, window_length)
        output[:, start:start + window_length, :] += x_windowed[:, i, :, :].permute(0, 2, 1)

    # Trim rise/fall of first and last window to recover exactly n_timesteps
    return output[:, hop_size:-hop_size, :]
