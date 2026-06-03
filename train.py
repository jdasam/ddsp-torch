#!/usr/bin/env python3
"""Train a DDSP model on preprocessed audio data."""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import soundfile as sf

from ddsp.model import DDSP
from ddsp.dataset import DDSPDataset
from ddsp.losses import spectral_loss
from ddsp.utils import get_scheduler


def compute_loudness_stats(dataloader):
    mean = std = n = 0.0
    for _, _, l in dataloader:
        n += 1
        mean += (l.mean().item() - mean) / n
        std  += (l.std().item()  - std)  / n
    return mean, std


def _save_checkpoint(path, model, opt, step, mean_loudness, std_loudness, cfg):
    torch.save({
        "model":         model.state_dict(),
        "optimizer":     opt.state_dict(),
        "step":          step,
        "mean_loudness": mean_loudness,
        "std_loudness":  std_loudness,
        "config":        cfg,
    }, str(path))


def train(config_path: str, name: str = "run", out: str = "runs", steps: int = None):
    """Train a DDSP model. Resumes automatically from `<out>/<name>/checkpoint.pt`
    if one exists, so session interruptions are recoverable.

    Args:
        config_path: path to YAML config (e.g. configs/default.yaml).
        name:   subdirectory under `out/` to save checkpoint and previews.
        out:    parent directory for run artifacts.
        steps:  total optimization steps. Falls back to cfg["steps"] when None.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    steps = steps or cfg["steps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(out) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    ckpt_path = run_dir / "checkpoint.pt"

    dataset    = DDSPDataset(cfg["out_dir"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    # Pick a fixed preview clip (highest voiced ratio) so previews across steps
    # are directly comparable.
    voiced_per_clip = (dataset.pitches > 0).sum(axis=1)
    preview_idx = int(voiced_per_clip.argmax())
    preview_sig, preview_pitch, preview_loud = dataset[preview_idx]
    preview_pitch_t = preview_pitch.unsqueeze(0).unsqueeze(-1).to(device)
    preview_loud_t  = preview_loud.unsqueeze(0).unsqueeze(-1).to(device)

    target_path = run_dir / "preview_target.wav"
    if not target_path.exists():
        sf.write(str(target_path),
                 preview_sig.numpy(), cfg["sampling_rate"])
        print(f"Preview reference clip: index={preview_idx}, "
              f"voiced={voiced_per_clip[preview_idx]}/{dataset.pitches.shape[1]} frames")

    model = DDSP(
        hidden_size=cfg["hidden_size"],
        n_harmonics=cfg["n_harmonics"],
        n_noise_bands=cfg["n_noise_bands"],
        sampling_rate=cfg["sampling_rate"],
        block_size=cfg["block_size"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["start_lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, get_scheduler(cfg["start_lr"], cfg["stop_lr"], cfg["decay_over"])
    )

    # ── Resume or start fresh ──────────────────────────────────────────────
    start_step = 0
    if ckpt_path.exists():
        print(f"Found existing checkpoint at {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step    = int(ckpt["step"])
        mean_loudness = float(ckpt["mean_loudness"])
        std_loudness  = float(ckpt["std_loudness"])
        for _ in range(start_step):
            scheduler.step()
        print(f"Resumed from step {start_step:,}")
    else:
        mean_loudness, std_loudness = compute_loudness_stats(dataloader)
        print(f"Loudness stats — mean: {mean_loudness:.2f}, std: {std_loudness:.2f}")

    if start_step >= steps:
        print(f"Checkpoint step ({start_step:,}) is already ≥ target ({steps:,}). "
              f"Nothing to do. Increase STEPS to continue training.")
        return

    # ── Training loop ──────────────────────────────────────────────────────
    step = start_step
    ema_loss = None
    preview_every = cfg.get("preview_every", 5000)
    pbar = tqdm(total=steps, initial=start_step, desc="Training", unit="step")

    done = False
    while not done:
        for signal, pitch, loudness in dataloader:
            signal   = signal.to(device)
            pitch    = pitch.unsqueeze(-1).to(device)
            loudness = loudness.unsqueeze(-1).to(device)
            loudness = (loudness - mean_loudness) / (std_loudness + 1e-7)

            pred = model(pitch, loudness).squeeze(-1)
            loss = spectral_loss(signal, pred, cfg["scales"], cfg["overlap"])

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            step += 1

            cur = loss.item()
            ema_loss = cur if ema_loss is None else 0.98 * ema_loss + 0.02 * cur
            pbar.update(1)
            pbar.set_postfix(loss=f"{ema_loss:.3f}",
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")

            if step % preview_every == 0:
                model.eval()
                with torch.no_grad():
                    loud_norm = (preview_loud_t - mean_loudness) / (std_loudness + 1e-7)
                    pred_preview = model(preview_pitch_t, loud_norm).squeeze(-1).squeeze(0)
                sf.write(str(run_dir / f"preview_{step:07d}.wav"),
                         pred_preview.cpu().numpy(), cfg["sampling_rate"])
                _save_checkpoint(ckpt_path, model, opt, step,
                                 mean_loudness, std_loudness, cfg)
                model.train()

            if step >= steps:
                done = True
                break

    pbar.close()
    _save_checkpoint(ckpt_path, model, opt, step,
                     mean_loudness, std_loudness, cfg)
    print(f"Checkpoint saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name",   default="run")
    parser.add_argument("--out",    default="runs")
    parser.add_argument("--steps",  type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.name, args.out, args.steps)


if __name__ == "__main__":
    main()
