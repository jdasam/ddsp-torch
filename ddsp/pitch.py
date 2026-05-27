import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
# Minimum fraction of voiced frames required to accept an extractor's output.
# If an extractor passes confidence_threshold for fewer than MIN_VOICED_RATIO
# frames, we fall through to the next extractor.
MIN_VOICED_RATIO = 0.2

# Pitch search range (Hz) — covers C2 (~65) up to ~2 kHz, well above violin/flute top.
FMIN = 50.0
FMAX = 2000.0


def _trim_or_pad(arr: np.ndarray, n_frames: int) -> np.ndarray:
    if len(arr) >= n_frames:
        return arr[:n_frames]
    return np.pad(arr, (0, n_frames - len(arr)))


def _with_torchcrepe(signal: np.ndarray, sr: int, block_size: int):
    """CREPE pitch via torchcrepe (PyTorch port, same pre-trained weights)."""
    import torch
    import torchcrepe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.from_numpy(signal).unsqueeze(0).to(device)
    pitch, periodicity = torchcrepe.predict(
        audio, sr, block_size, FMIN, FMAX,
        model="full",
        return_periodicity=True,
        device=device,
        batch_size=512,
    )
    return (
        pitch.squeeze(0).cpu().numpy().astype(np.float32),
        periodicity.squeeze(0).cpu().numpy().astype(np.float32),
    )


def _with_pyin(signal: np.ndarray, sr: int, block_size: int):
    import librosa

    f0, _voiced_flag, voiced_probs = librosa.pyin(
        signal,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        hop_length=block_size,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    conf = np.nan_to_num(voiced_probs, nan=0.0).astype(np.float32)
    return f0, conf


def extract_pitch(
    signal: np.ndarray,
    sampling_rate: int,
    block_size: int,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Extract f0 using torchcrepe -> PYIN fallback chain.

    Frames whose voicing confidence is below `confidence_threshold` are zeroed
    out (treated as unvoiced). If an extractor's voiced ratio is below
    MIN_VOICED_RATIO, it is skipped and the next extractor is tried — this
    guards against extractor-specific calibration issues on certain instruments.

    Args:
        signal:        (n_samples,) float32 mono audio
        sampling_rate: sample rate in Hz
        block_size:    hop size = samples per frame
        confidence_threshold: in [0, 1]; frames below this are set to 0.

    Returns:
        f0: (n_frames,) float32 in Hz, 0 where unvoiced / low confidence
    """
    n_frames = len(signal) // block_size

    for name, fn in [
        ("torchcrepe", _with_torchcrepe),
        ("PYIN",       _with_pyin),
    ]:
        try:
            f0, conf = fn(signal, sampling_rate, block_size)
            f0 = _trim_or_pad(f0, n_frames)
            conf = _trim_or_pad(conf, n_frames)
            voiced_ratio = (conf >= confidence_threshold).mean() if confidence_threshold > 0.0 else 1.0
            if voiced_ratio < MIN_VOICED_RATIO:
                logger.warning(
                    "%s voiced ratio %.1f%% < %.0f%% — falling through to next extractor",
                    name, voiced_ratio * 100, MIN_VOICED_RATIO * 100,
                )
                continue
            if confidence_threshold > 0.0:
                f0 = np.where(conf >= confidence_threshold, f0, 0.0).astype(np.float32)
            logger.info("Pitch extracted with %s (voiced=%.1f%%)", name, voiced_ratio * 100)
            return f0
        except Exception as exc:
            logger.warning("%s failed (%s), trying next extractor", name, exc)

    logger.error("All extractors failed — returning zeros")
    return np.zeros(n_frames, dtype=np.float32)
