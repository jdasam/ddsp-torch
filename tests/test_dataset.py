import os
import tempfile
import numpy as np
import torch
from ddsp.dataset import preprocess_audio, save_preprocessed, load_preprocessed, DDSPDataset


def _fake_audio(sr=16000, duration=8.0):
    return (np.random.randn(int(sr * duration)) * 0.1).astype(np.float32)


def test_preprocess_shapes():
    audio = _fake_audio(duration=4.0)
    signals, pitches, loudnesses = preprocess_audio(audio, 16000, 160, 16000)
    n_clips = 4        # 4s audio / 1s clip
    n_frames = 16000 // 160
    assert signals.shape  == (n_clips, 16000)
    assert pitches.shape  == (n_clips, n_frames)
    assert loudnesses.shape == (n_clips, n_frames)


def test_save_and_load():
    audio = _fake_audio(duration=4.0)
    signals, pitches, loudnesses = preprocess_audio(audio, 16000, 160, 16000)
    with tempfile.TemporaryDirectory() as d:
        save_preprocessed(d, signals, pitches, loudnesses)
        assert os.path.exists(os.path.join(d, "signals.npy"))
        s2, p2, l2 = load_preprocessed(d)
        np.testing.assert_array_equal(signals, s2)


def test_dataset_getitem():
    audio = _fake_audio(duration=8.0)
    signals, pitches, loudnesses = preprocess_audio(audio, 16000, 160, 16000)
    with tempfile.TemporaryDirectory() as d:
        save_preprocessed(d, signals, pitches, loudnesses)
        ds = DDSPDataset(d)
        assert len(ds) == signals.shape[0]
        s, p, l = ds[0]
        assert isinstance(s, torch.Tensor)
        assert s.dtype == torch.float32
