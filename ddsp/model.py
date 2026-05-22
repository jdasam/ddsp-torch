import math
import torch
import torch.nn as nn
from .core import (
    harmonic_synth, filtered_noise, fft_convolve,
    remove_above_nyquist, upsample, upsample_with_windows,
)


def mlp(in_size: int, hidden_size: int, n_layers: int) -> nn.Sequential:
    """MLP with LayerNorm + LeakyReLU activations."""
    layers: list = [nn.Linear(in_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU()]
    return nn.Sequential(*layers)


def scale_function(x: torch.Tensor) -> torch.Tensor:
    """Map any real input to strictly positive output. Range: (1e-7, ~2)."""
    return 2 * torch.sigmoid(x) ** math.log(10) + 1e-7


class Reverb(nn.Module):
    def __init__(self, length: int, sampling_rate: int,
                 initial_wet: float = 0.0, initial_decay: float = 5.0):
        super().__init__()
        self.length = length
        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet   = nn.Parameter(torch.tensor(float(initial_wet)))
        t = (torch.arange(length) / sampling_rate).reshape(1, -1, 1)
        self.register_buffer("t", t)

    def _build_ir(self) -> torch.Tensor:
        envelope = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        ir = self.noise * envelope * torch.sigmoid(self.wet)
        ir[:, 0] = 1                                    # preserve direct signal
        return ir                                        # (1, length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable reverb IR via FFT convolution.

        Args:
            x: (batch, n_samples, 1)
        Returns:
            (batch, n_samples, 1)
        """
        ir = self._build_ir()                            # (1, length, 1)
        ir = nn.functional.pad(ir, (0, 0, 0, x.shape[1] - self.length))
        # fft_convolve operates on last dim; squeeze/unsqueeze channel dim
        out = fft_convolve(x.squeeze(-1), ir.squeeze(-1))
        return out.unsqueeze(-1)


class DDSP(nn.Module):
    def __init__(self, hidden_size: int, n_harmonics: int, n_noise_bands: int,
                 sampling_rate: int, block_size: int,
                 use_angular_cumsum: bool = False):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size",    torch.tensor(block_size))
        self.n_harmonics   = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.use_angular_cumsum = use_angular_cumsum

        self.pitch_mlp    = mlp(1, hidden_size, 3)
        self.loudness_mlp = mlp(1, hidden_size, 3)
        self.gru          = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out_mlp      = mlp(hidden_size + 2, hidden_size, 3)

        self.harmonic_proj = nn.Linear(hidden_size, n_harmonics + 1)  # total_amp + n_harmonics
        self.noise_proj    = nn.Linear(hidden_size, n_noise_bands)

        self.reverb = Reverb(sampling_rate, sampling_rate)

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor) -> torch.Tensor:
        """Synthesize audio from frame-rate pitch and loudness.

        Args:
            pitch:    (batch, n_frames, 1) f0 in Hz
            loudness: (batch, n_frames, 1) normalized A-weighted loudness

        Returns:
            audio: (batch, n_frames * block_size, 1)
        """
        sr = self.sampling_rate.item()
        bs = self.block_size.item()

        # Encode
        hidden = torch.cat([self.pitch_mlp(pitch), self.loudness_mlp(loudness)], dim=-1)
        hidden, _ = self.gru(hidden)
        hidden = self.out_mlp(torch.cat([hidden, pitch, loudness], dim=-1))

        # Harmonic branch
        h_params = scale_function(self.harmonic_proj(hidden))
        total_amp  = h_params[..., :1]
        amplitudes = h_params[..., 1:]
        amplitudes = remove_above_nyquist(amplitudes, pitch, sr)
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-7) * total_amp
        n_samples = pitch.shape[1] * bs
        harmonic = harmonic_synth(
            upsample(pitch, bs),
            upsample_with_windows(amplitudes, n_samples),
            sr,
            use_angular_cumsum=self.use_angular_cumsum,
        )

        # Noise branch
        n_params = scale_function(self.noise_proj(hidden) - 5)
        noise = filtered_noise(n_params, bs)

        return self.reverb(harmonic + noise)
