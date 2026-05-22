import numpy as np
from ddsp.pitch import extract_pitch


def _sine(freq, sr=16000, duration=2.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_shape():
    signal = _sine(440.0)
    pitch = extract_pitch(signal, sampling_rate=16000, block_size=160)
    assert pitch.shape == (16000 * 2 // 160,)


def test_dtype():
    signal = _sine(440.0)
    pitch = extract_pitch(signal, 16000, 160)
    assert pitch.dtype == np.float32


def test_approximate_frequency():
    # Pure 440 Hz sine — pitch should be near 440 (±50 Hz tolerance across extractors)
    signal = _sine(440.0, duration=3.0)
    pitch = extract_pitch(signal, 16000, 160)
    voiced = pitch[pitch > 0]
    assert len(voiced) > 0
    assert 390.0 < voiced.mean() < 490.0
