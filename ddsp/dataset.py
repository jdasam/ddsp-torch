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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice audio into fixed-length clips; extract pitch and loudness per clip.

    Args:
        audio:         (n_samples,) float32 mono
        sampling_rate: Hz
        block_size:    samples per frame
        signal_length: samples per training clip
        oneshot:       if True, use only the first clip

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
