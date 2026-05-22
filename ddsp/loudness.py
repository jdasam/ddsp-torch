import numpy as np
import librosa


def extract_loudness(
    signal: np.ndarray,
    sampling_rate: int,
    block_size: int,
    n_fft: int = 2048,
) -> np.ndarray:
    """A-weighted RMS loudness in dB, one value per frame.

    Args:
        signal:       (n_samples,) float32 mono audio
        sampling_rate: sample rate in Hz
        block_size:   hop size = samples per frame
        n_fft:        FFT window size for STFT

    Returns:
        loudness: (n_frames,) float32 in log-energy units
    """
    S = np.abs(librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    ))

    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    freqs[0] = freqs[1]                             # avoid log(0) for DC bin
    a_weights = librosa.A_weighting(freqs)          # dB A-weighting per bin

    # Convert magnitude to log scale and apply A-weighting
    S_log = np.log(S + 1e-7) + a_weights[:, np.newaxis] * (np.log(10) / 20)
    loudness = np.mean(S_log, axis=0)               # average over freq → (n_frames,)

    n_frames = len(signal) // block_size
    return loudness[:n_frames].astype(np.float32)
