#!/usr/bin/env python3
"""Timbre transfer: synthesize audio in the style of a trained DDSP model."""
import argparse

import numpy as np
import torch
import soundfile as sf
import librosa

from ddsp.model import DDSP
from ddsp.pitch import extract_pitch
from ddsp.loudness import extract_loudness


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = DDSP(
        hidden_size=cfg["hidden_size"],
        n_harmonics=cfg["n_harmonics"],
        n_noise_bands=cfg["n_noise_bands"],
        sampling_rate=cfg["sampling_rate"],
        block_size=cfg["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.use_angular_cumsum = True  # prevent phase drift on long sequences
    model.eval()
    return model, cfg, ckpt["mean_loudness"], ckpt["std_loudness"]


def timbre_transfer(
    source_path: str,
    checkpoint_path: str,
    output_path: str,
    pitch_shift_semitones: float = 0.0,
    loudness_db_offset: float = 0.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, mean_l, std_l = load_checkpoint(checkpoint_path, device)

    sr = cfg["sampling_rate"]
    bs = cfg["block_size"]

    audio, source_sr = sf.read(source_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    f0       = extract_pitch(audio, sr, bs)
    loudness = extract_loudness(audio, sr, bs)

    if pitch_shift_semitones:
        f0 = f0 * (2 ** (pitch_shift_semitones / 12))
    if loudness_db_offset:
        loudness = loudness + loudness_db_offset

    loudness_norm = (loudness - mean_l) / (std_l + 1e-7)

    pitch_t    = torch.from_numpy(f0).unsqueeze(0).unsqueeze(-1).to(device)
    loudness_t = torch.from_numpy(loudness_norm).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        out = model(pitch_t, loudness_t).squeeze().cpu().numpy()

    sf.write(output_path, out, sr)
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="DDSP Timbre Transfer")
    p.add_argument("--source",          required=True)
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--output",          required=True)
    p.add_argument("--pitch-shift",     type=float, default=0.0, help="Semitones (±12)")
    p.add_argument("--loudness-offset", type=float, default=0.0, help="dB offset (±20)")
    args = p.parse_args()
    timbre_transfer(args.source, args.checkpoint, args.output,
                    args.pitch_shift, args.loudness_offset)


if __name__ == "__main__":
    main()
