#!/usr/bin/env python3
"""Generate the DDSP Colab demo notebook."""
import json
import pathlib

pathlib.Path("notebooks").mkdir(exist_ok=True)


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("""# DDSP PyTorch — Timbre Transfer

Train a DDSP model on your own instrument audio, then apply **timbre transfer**:
any melody played on any instrument gets resynthesized with the trained timbre.

**Before you start:** Runtime → Change runtime type → **T4 GPU**

**Training audio requirements:** 10–20 min, mono, single acoustic instrument
(e.g. violin, flute, cello). Recording quality matters more than quantity.

> Dataset tip: Individual instrument WAV tracks from the
> [URMP dataset](https://goo.gl/forms/xSvMzlwl3IWijvcp2) work great.
> Fill in the form to get the download link, then use the individual
> instrument WAV files (e.g. `AuSep_1_vn_01_Spring.wav` for violin).
"""),

    code("""\
# @title 1. Clone repo and install environment
import subprocess, sys, os

REPO_URL = "https://github.com/YOUR_USERNAME/ddsp-pytorch.git"  # update before sharing
if not os.path.exists("ddsp-pytorch"):
    subprocess.run(["git", "clone", REPO_URL, "ddsp-pytorch"], check=True)
os.chdir("ddsp-pytorch")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
# Note: crepe requires pip (not uv) due to legacy build system
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "crepe>=0.0.14"], check=True)
subprocess.run(["uv", "pip", "install", "--system", "-q", "-e", "."], check=True)
print("✓ Installation complete")
"""),

    code("""\
# @title 2. Check GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected — training will be very slow!")
"""),

    code("""\
# @title 3. Upload training audio
# @markdown Upload a WAV or MP3 file.
# @markdown **Aim for 10–20 minutes of mono, single-instrument audio.**
# @markdown Tip: URMP dataset individual tracks work great.
from google.colab import files
import os

uploaded = files.upload()
AUDIO_FILE = list(uploaded.keys())[0]
print(f"Uploaded: {AUDIO_FILE}  ({os.path.getsize(AUDIO_FILE)/1e6:.1f} MB)")
"""),

    code("""\
# @title 4. Preprocess audio (extract pitch + loudness)
# @markdown This may take a few minutes depending on audio length.
# @markdown Pitch is extracted with PESTO (fast, PyTorch-based).
import numpy as np, librosa
from ddsp.dataset import preprocess_audio, save_preprocessed

SR          = 16000
BLOCK_SIZE  = 160
SIGNAL_LEN  = 64000   # 4-second training clips
OUT_DIR     = "./preprocessed"

audio, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True)
n_clips = len(audio) // SIGNAL_LEN
print(f"Duration: {len(audio)/SR:.1f}s → {n_clips} training clips of 4s each")

if n_clips < 4:
    print("⚠ Warning: very few clips. Consider using longer audio (10+ minutes).")

print("Extracting pitch and loudness ...")
signals, pitches, loudnesses = preprocess_audio(audio, SR, BLOCK_SIZE, SIGNAL_LEN)
save_preprocessed(OUT_DIR, signals, pitches, loudnesses)
print(f"✓ {signals.shape[0]} clips saved to {OUT_DIR}/")
"""),

    code("""\
# @title 5. Train model
# @markdown Adjust settings, then run. A 50k-step run takes ~45–60 min on T4.
import subprocess, sys

STEPS      = 50000  # @param {type:"integer"}
BATCH_SIZE = 16     # @param {type:"integer"}
RUN_NAME   = "colab_run"

# Patch config with chosen batch size
import yaml
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["batch_size"] = BATCH_SIZE
cfg["out_dir"] = OUT_DIR
with open("configs/colab.yaml", "w") as f:
    yaml.safe_dump(cfg, f)

result = subprocess.run([
    sys.executable, "train.py",
    "--config", "configs/colab.yaml",
    "--name",   RUN_NAME,
    "--steps",  str(STEPS),
], capture_output=False)
print(f"\\n✓ Training complete. Checkpoint at runs/{RUN_NAME}/checkpoint.pt")
"""),

    code("""\
# @title 6. Upload source audio for timbre transfer
# @markdown Upload any audio you want to transform (WAV or MP3).
# @markdown Short clips (10-30 seconds) work well for testing.
from google.colab import files
uploaded_src = files.upload()
SOURCE_FILE = list(uploaded_src.keys())[0]
print(f"Source: {SOURCE_FILE}")
"""),

    code("""\
# @title 7. Timbre Transfer
# @markdown Adjust parameters and run the cell.
PITCH_SHIFT     =  0    # @param {type:"slider", min:-12, max:12,  step:0.5}
LOUDNESS_OFFSET =  0    # @param {type:"slider", min:-20, max:20,  step:1}
OUTPUT_FILE     = "output.wav"

from inference import timbre_transfer
timbre_transfer(
    source_path           = SOURCE_FILE,
    checkpoint_path       = f"runs/colab_run/checkpoint.pt",
    output_path           = OUTPUT_FILE,
    pitch_shift_semitones = PITCH_SHIFT,
    loudness_db_offset    = LOUDNESS_OFFSET,
)
print("✓ Done!")
"""),

    code("""\
# @title 8. Listen and download
import soundfile as sf
from IPython.display import Audio, display
from google.colab import files

print("▶ Source audio:")
src, src_sr = sf.read(SOURCE_FILE)
display(Audio(src, rate=src_sr))

print("▶ Timbre transfer result:")
out, out_sr = sf.read(OUTPUT_FILE)
display(Audio(out, rate=out_sr))

files.download(OUTPUT_FILE)
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "accelerator": "GPU",
        "colab": {"name": "DDSP PyTorch Timbre Transfer"},
    },
    "cells": cells,
}

with open("notebooks/colab_demo.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("✓ notebooks/colab_demo.ipynb written")
