import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from .pitch import extract_pitch
from .loudness import extract_loudness


def preprocess_audio(
    audio: np.ndarray,
    sampling_rate: int,
    block_size: int,
    signal_length: int,
    oneshot: bool = False,
    min_voiced_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice audio into fixed-length clips; extract pitch and loudness per clip.

    Clips whose voiced-frame ratio (pitch > 0) is below `min_voiced_ratio` are
    dropped — they carry no timbre signal for the harmonic synthesizer to learn
    and would also bias the loudness statistics. Pass `min_voiced_ratio=0` to
    keep every clip.

    Args:
        audio:            (n_samples,) float32 mono
        sampling_rate:    Hz
        block_size:       samples per frame
        signal_length:    samples per training clip
        oneshot:          if True, use only the first clip
        min_voiced_ratio: drop clips below this voiced fraction (default 0.05)

    Returns:
        signals:   (n_clips, signal_length) float32
        pitches:   (n_clips, n_frames) float32 Hz
        loudnesses:(n_clips, n_frames) float32
    """
    remainder = len(audio) % signal_length
    if remainder:
        audio = np.pad(audio, (0, signal_length - remainder))
    if oneshot:
        audio = audio[:signal_length]

    n_clips = len(audio) // signal_length
    signals = audio[:n_clips * signal_length].reshape(n_clips, signal_length)
    pitches    = np.stack([extract_pitch(signals[i], sampling_rate, block_size)   for i in range(n_clips)])
    loudnesses = np.stack([extract_loudness(signals[i], sampling_rate, block_size) for i in range(n_clips)])

    if min_voiced_ratio > 0.0:
        voiced_ratio = (pitches > 0).mean(axis=1)
        keep = voiced_ratio >= min_voiced_ratio
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(f"Dropping {n_dropped} / {n_clips} clip(s) with voiced ratio < "
                  f"{min_voiced_ratio*100:.0f}% (silent / no pitched content)")
        signals    = signals[keep]
        pitches    = pitches[keep]
        loudnesses = loudnesses[keep]

    return signals, pitches, loudnesses


def save_preprocessed(
    out_dir: str,
    signals: np.ndarray,
    pitches: np.ndarray,
    loudnesses: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "signals.npy"),  signals)
    np.save(os.path.join(out_dir, "pitchs.npy"),   pitches)
    np.save(os.path.join(out_dir, "loudness.npy"), loudnesses)


def load_preprocessed(out_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(os.path.join(out_dir, "signals.npy")),
        np.load(os.path.join(out_dir, "pitchs.npy")),
        np.load(os.path.join(out_dir, "loudness.npy")),
    )


class DDSPDataset(Dataset):
    def __init__(self, out_dir: str):
        self.signals, self.pitches, self.loudnesses = load_preprocessed(out_dir)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.pitches[idx]),
            torch.from_numpy(self.loudnesses[idx]),
        )
