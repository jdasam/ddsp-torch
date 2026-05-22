from typing import List
import torch
from .core import safe_log


def multiscale_fft(
    signal: torch.Tensor,
    scales: List[int],
    overlap: float,
) -> List[torch.Tensor]:
    """STFT magnitude spectrograms at multiple FFT sizes.

    Args:
        signal:  (batch, n_samples)
        scales:  list of FFT sizes (n_fft values)
        overlap: hop overlap fraction, e.g. 0.75 → hop = scale * 0.25

    Returns:
        list of (batch, n_freq, n_frames) magnitude spectrograms
    """
    stfts = []
    for s in scales:
        hop = int(s * (1 - overlap))
        win = torch.hann_window(s, device=signal.device, dtype=signal.dtype)
        S = torch.stft(
            signal,
            n_fft=s,
            hop_length=hop,
            win_length=s,
            window=win,
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def spectral_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    scales: List[int],
    overlap: float,
) -> torch.Tensor:
    """Multi-scale spectral loss: L1 on linear magnitude + L1 on log magnitude.

    Args:
        target:     (batch, n_samples) ground truth audio
        prediction: (batch, n_samples) synthesized audio
        scales:     list of FFT sizes
        overlap:    STFT hop overlap fraction

    Returns:
        scalar loss tensor
    """
    ori = multiscale_fft(target, scales, overlap)
    rec = multiscale_fft(prediction, scales, overlap)
    loss = torch.tensor(0.0, device=target.device)
    for s_x, s_y in zip(ori, rec):
        loss = loss + (s_x - s_y).abs().mean()
        loss = loss + (safe_log(s_x) - safe_log(s_y)).abs().mean()
    return loss
