import numpy as np
import librosa


DB_RANGE = 80.0  # matches Magenta core.DB_RANGE


def extract_loudness(
    signal: np.ndarray,
    sampling_rate: int,
    block_size: int,
    n_fft: int = 2048,
    ref_db: float = 0.0,
    range_db: float = DB_RANGE,
) -> np.ndarray:
    """Perceptual A-weighted loudness in dB (Magenta-style).

    Matches the recipe in ddsp/ddsp/spectral_ops.py compute_loudness:
      1. STFT → power spectrum
      2. Multiply by A-weighting in linear power scale (10^(A_db / 10))
      3. Mean across frequency bins (perceptual weighted power per frame)
      4. Convert to dB with floor at -range_db
      5. Subtract ref_db (==0 by default in current Magenta)

    Args:
        signal:        (n_samples,) float32 mono audio
        sampling_rate: sample rate in Hz
        block_size:    hop size = samples per frame
        n_fft:         FFT window size
        ref_db:        reference loudness shift in dB (Magenta default: 0.0)
        range_db:      dynamic-range floor in dB (Magenta default: 80.0)

    Returns:
        loudness: (n_frames,) float32 dB values in [-range_db, +inf)
    """
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    power = np.abs(S) ** 2

    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    freqs[0] = freqs[1]                                          # avoid log(0) for DC bin
    weighting = 10.0 ** (librosa.A_weighting(freqs) / 10.0)      # dB → power scale
    power = power * weighting[:, np.newaxis]

    avg_power = np.mean(power, axis=0)                           # mean across freq → (n_frames,)

    pmin = 10.0 ** (-range_db / 10.0)
    avg_power = np.maximum(pmin, avg_power)
    db = 10.0 * np.log10(avg_power) - ref_db
    db = np.maximum(db, -range_db)

    n_frames = len(signal) // block_size
    return db[:n_frames].astype(np.float32)
