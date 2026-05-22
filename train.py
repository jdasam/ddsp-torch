#!/usr/bin/env python3
"""Train a DDSP model on preprocessed audio data."""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name",   default="run")
    parser.add_argument("--out",    default="runs")
    parser.add_argument("--steps",  type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    steps = args.steps or cfg["steps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(args.out) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    dataset    = DDSPDataset(cfg["out_dir"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    mean_loudness, std_loudness = compute_loudness_stats(dataloader)

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

    step = 0
    epochs = int(np.ceil(steps / len(dataloader)))

    for _ in tqdm(range(epochs), desc="Epochs"):
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

            if step % cfg.get("log_every", 500) == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {step}/{steps}  loss={loss.item():.4f}  lr={lr:.2e}")

            if step % cfg.get("preview_every", 5000) == 0:
                model.eval()
                with torch.no_grad():
                    sf.write(str(run_dir / f"preview_{step:07d}.wav"),
                             pred[0].cpu().numpy(), cfg["sampling_rate"])
                torch.save({
                    "model": model.state_dict(), "step": step,
                    "mean_loudness": mean_loudness, "std_loudness": std_loudness,
                    "config": cfg,
                }, str(run_dir / "checkpoint.pt"))
                model.train()

            if step >= steps:
                break

    torch.save({
        "model": model.state_dict(), "step": step,
        "mean_loudness": mean_loudness, "std_loudness": std_loudness,
        "config": cfg,
    }, str(run_dir / "checkpoint.pt"))
    print(f"Checkpoint saved to {run_dir}/checkpoint.pt")


if __name__ == "__main__":
    main()
