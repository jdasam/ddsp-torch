import numpy as np
from ddsp.loudness import extract_loudness


def test_shape():
    signal = np.random.randn(16000).astype(np.float32) * 0.1
    loudness = extract_loudness(signal, sampling_rate=16000, block_size=160)
    assert loudness.shape == (16000 // 160,)


def test_dtype():
    signal = np.zeros(16000, dtype=np.float32)
    loudness = extract_loudness(signal, 16000, 160)
    assert loudness.dtype == np.float32


def test_louder_signal_is_louder():
    sr, bs, n = 16000, 160, 16000
    quiet = np.random.randn(n).astype(np.float32) * 0.01
    loud = np.random.randn(n).astype(np.float32) * 1.0
    assert extract_loudness(loud, sr, bs).mean() > extract_loudness(quiet, sr, bs).mean()
