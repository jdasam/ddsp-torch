import logging
import numpy as np

logger = logging.getLogger(__name__)


def _trim_or_pad(arr: np.ndarray, n_frames: int) -> np.ndarray:
    if len(arr) >= n_frames:
        return arr[:n_frames]
    return np.pad(arr, (0, n_frames - len(arr)))


def _with_pesto(signal: np.ndarray, sr: int, block_size: int) -> np.ndarray:
    import torch
    import pesto

    audio = torch.from_numpy(signal).unsqueeze(0)       # (1, n_samples)
    step_ms = block_size / sr * 1000
    _, f0, _, _ = pesto.predict(audio, sr, step_size=step_ms, convert_to_freq=True)
    # f0 has shape (batch_size, num_timesteps) when input is batched
    if f0.ndim == 2:
        f0 = f0.squeeze(0)
    return f0.numpy().astype(np.float32)


def _with_crepe(signal: np.ndarray, sr: int, block_size: int) -> np.ndarray:
    import crepe

    step_ms = block_size / sr * 1000
    _, frequency, _, _ = crepe.predict(
        signal, sr, step_size=step_ms, viterbi=True, verbose=0
    )
    return frequency.astype(np.float32)


def _with_pyin(signal: np.ndarray, sr: int, block_size: int) -> np.ndarray:
    import librosa

    f0, _, _ = librosa.pyin(
        signal,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=block_size,
    )
    return np.nan_to_num(f0, nan=0.0).astype(np.float32)


def extract_pitch(
    signal: np.ndarray,
    sampling_rate: int,
    block_size: int,
) -> np.ndarray:
    """Extract f0 using PESTO -> CREPE -> PYIN fallback chain.

    Args:
        signal:       (n_samples,) float32 mono audio
        sampling_rate: sample rate in Hz
        block_size:   hop size = samples per frame

    Returns:
        f0: (n_frames,) float32 in Hz, 0 where unvoiced
    """
    n_frames = len(signal) // block_size

    for name, fn in [
        ("PESTO", _with_pesto),
        ("CREPE", _with_crepe),
        ("PYIN",  _with_pyin),
    ]:
        try:
            f0 = fn(signal, sampling_rate, block_size)
            f0 = _trim_or_pad(f0, n_frames)
            logger.debug("Pitch extracted with %s", name)
            return f0
        except Exception as exc:
            logger.warning("%s failed (%s), trying next extractor", name, exc)

    logger.error("All extractors failed — returning zeros")
    return np.zeros(n_frames, dtype=np.float32)
