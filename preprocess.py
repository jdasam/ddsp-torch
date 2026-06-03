#!/usr/bin/env python3
"""Extract per-frame pitch and loudness from an audio file and save as .npy."""
import argparse
from pathlib import Path

import librosa

from ddsp.dataset import preprocess_audio, save_preprocessed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True,
                   help="Path to input audio file (WAV / MP3 / OGG / FLAC).")
    p.add_argument("--out", default="./preprocessed/train",
                   help="Output directory for .npy files (matches default config).")
    p.add_argument("--sr",            type=int, default=16000)
    p.add_argument("--block-size",    type=int, default=160)
    p.add_argument("--signal-length", type=int, default=64000)
    p.add_argument("--min-voiced-ratio", type=float, default=0.05,
                   help="Drop clips with fewer voiced frames than this fraction "
                        "(default 0.05). Pass 0 to keep every clip.")
    args = p.parse_args()

    src = Path(args.audio)
    print(f"Loading {src} at {args.sr} Hz...")
    audio, _ = librosa.load(str(src), sr=args.sr, mono=True)
    dur = len(audio) / args.sr
    n_clips = len(audio) // args.signal_length
    print(f"Duration: {dur:.1f}s ({dur/60:.1f} min)  →  {n_clips} clips of "
          f"{args.signal_length/args.sr:.0f}s each")

    print("Extracting pitch & loudness ...")
    signals, pitches, loudnesses = preprocess_audio(
        audio, args.sr, args.block_size, args.signal_length,
        min_voiced_ratio=args.min_voiced_ratio,
    )
    save_preprocessed(args.out, signals, pitches, loudnesses)
    print(f"✓ Saved {signals.shape[0]} clips to {args.out}/")


if __name__ == "__main__":
    main()
