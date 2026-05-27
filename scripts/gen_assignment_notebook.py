#!/usr/bin/env python3
"""Generate the DDSP Assignment notebook: Build Your Own Timbre Model."""
import json
import pathlib

pathlib.Path("notebooks").mkdir(exist_ok=True)


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

cells = [

# ── Title ──────────────────────────────────────────────────────────────────
md("""\
# DDSP Assignment: Build Your Own Timbre Model

**DDSP** (Differentiable Digital Signal Processing) learns to synthesize the *timbre* of an \
instrument from raw audio. Once trained, it performs **timbre transfer**: any melody played on \
any instrument gets resynthesized in your instrument's voice.

> **Your task:** Choose audio that is interesting and suitable, train a model on it, \
then explore what timbre transfer can do.
> You are **not** here to write code — all code is provided. \
Your job is to **listen carefully, experiment, and reflect**.

---

**Before you start:** `Runtime → Change runtime type → T4 GPU`
"""),

# ── Background ─────────────────────────────────────────────────────────────
md("""\
## Background: How DDSP Works

DDSP represents sound as two components synthesized in parallel:
- **Harmonic synthesizer** — a sum of sinusoids at integer multiples of the fundamental frequency
- **Noise synthesizer** — filtered noise for the inharmonic (noisy) part of the sound

A small neural network learns to control both from just two signals extracted from audio:

| Signal | What it encodes | Frame rate |
|--------|----------------|------------|
| **Pitch (F0)** | Fundamental frequency at each moment (Hz) | 100 frames/sec |
| **Loudness** | A-weighted acoustic energy | 100 frames/sec |

**Training:** The model sees (pitch, loudness) pairs and tries to reconstruct the original \
audio. It is optimized with a multi-scale spectral loss.

**Timbre transfer:** Extract pitch & loudness from any new audio → feed into the trained \
model → the output has the *melody* of the source and the *timbre* of the trained instrument.

```
[Source Audio] ──→ extract pitch & loudness ──→ [Trained DDSP Model] ──→ [Resynthesized Audio]
                                                         ↑
                                              learned from your training audio
```

Because only pitch and loudness are used, DDSP works best with **monophonic, \
sustained, pitched sounds**. Chords and percussion cannot be represented this way.
"""),

# ── 1. Setup ───────────────────────────────────────────────────────────────
code("""\
# @title 1. Setup — Run This First (takes ~1 minute)
# @markdown Installs dependencies and loads the DDSP code.
import subprocess, sys, os, warnings
warnings.filterwarnings("ignore")

REPO_URL = "https://github.com/jdasam/ddsp-torch.git"  # @param {type:"string"}

if not os.path.exists("ddsp-torch"):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, "ddsp-torch"], check=True)
os.chdir("ddsp-torch")

deps = ["soundfile", "librosa", "torchcrepe", "pyyaml", "einops", "tqdm"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
print("✓ Setup complete!")
"""),

# ── 2. GPU check ───────────────────────────────────────────────────────────
code("""\
# @title 2. Check GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✓ GPU available — training will be fast.")
else:
    print("⚠  No GPU detected!")
    print("Go to Runtime → Change runtime type → T4 GPU, then re-run all cells.")
"""),

# ── 3. Drive mount (optional) ──────────────────────────────────────────────
code("""\
# @title 3. (Optional) Mount Google Drive for Persistent Storage
# @markdown Recommended if you might pause and resume training across sessions.
# @markdown Skip this cell if you plan to complete everything in one session.
USE_DRIVE = False  # @param {type:"boolean"}

DRIVE_SAVE_DIR = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    import os
    DRIVE_SAVE_DIR = "/content/drive/MyDrive/DDSP_Assignment"
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    print(f"✓ Drive mounted. Checkpoints will save to: {DRIVE_SAVE_DIR}")
else:
    print("Drive not mounted. Checkpoints will be saved locally (lost on session reset).")
"""),

# ── Audio selection guide ──────────────────────────────────────────────────
md("""\
---
## Part 1: Your Training Audio

What you upload here is the most important decision in this assignment. \
DDSP learns the timbre of *one specific sound source* — your model will only be as good as \
the audio you feed it.

### What works well ✓

| Category | Examples |
|----------|---------|
| Solo acoustic instruments | Violin, cello, viola, flute, oboe, clarinet, trumpet, French horn |
| Singing / humming | Sustained vowels, a cappella melody, overtone singing |
| Unusual pitched sounds | Theremin, singing saw, glass harmonica, bowed metal |

**Key properties for good training audio:**
- **Monophonic** — one note at a time (no chords, no accompaniment)
- **Sustained** — notes held for at least half a second
- **Clear pitch** — the F0 estimator must be able to track the melody
- **Consistent timbre** — the same instrument and playing style throughout
- **Enough data** — at least 2 minutes, ideally 10–30 minutes

### What works poorly ✗
- Drums, cymbals, or other non-pitched percussion
- Piano chords or ensemble recordings (multiple simultaneous pitches)
- Highly staccato, rhythmic, or percussive playing style
- Speech with many unvoiced consonants
- Very noisy or heavily reverberant recordings

### Where to find suitable audio
- **Record yourself** — a phone microphone is fine; sing, hum, or play an instrument
- **[URMP dataset](https://goo.gl/forms/xSvMzlwl3IWijvcp2)** — fill the form to get a download link to isolated instrument tracks
- **Freesound.org** — search "solo violin sustained" or similar
- Your own existing recordings or samples

> **Minimum:** ~2 minutes (gives ~30 training clips of 4 seconds each)
> **Recommended:** 10–30 minutes for noticeably better quality
"""),

# ── 4. Upload audio ────────────────────────────────────────────────────────
code("""\
# @title 4. Upload Your Training Audio
# @markdown Upload WAV, MP3, OGG, or FLAC.
# @markdown If you have multiple files, upload them all — they will be concatenated.
from google.colab import files
import os

print("Select your training audio file(s):")
uploaded = files.upload()

TRAINING_FILES = list(uploaded.keys())
if TRAINING_FILES:
    total_mb = sum(os.path.getsize(f) for f in TRAINING_FILES) / 1e6
    print(f"\\nUploaded {len(TRAINING_FILES)} file(s), {total_mb:.1f} MB total:")
    for f in TRAINING_FILES:
        print(f"  • {f}  ({os.path.getsize(f) / 1e6:.1f} MB)")
else:
    print("No files uploaded.")
"""),

# ── 5. Inspect audio ───────────────────────────────────────────────────────
code("""\
# @title 5. Inspect Your Audio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display

SR = 16000  # DDSP target sample rate

# Load and concatenate all uploaded files
audio_parts = [librosa.load(f, sr=SR, mono=True)[0] for f in TRAINING_FILES]
audio = np.concatenate(audio_parts).astype(np.float32)

duration = len(audio) / SR
n_clips = int(len(audio) // 64000)  # 4-second clips

print(f"Total duration : {duration:.1f}s ({duration / 60:.1f} min)")
print(f"Training clips : {n_clips}  (4 seconds each)")

if n_clips < 10:
    print("⚠  Very few clips — model quality may be limited. More audio recommended.")
elif n_clips < 30:
    print("   Acceptable, but more audio would improve results.")
else:
    print("✓ Good amount of training data!")

# ── Visualize ──────────────────────────────────────────────────────────────
show_sec = min(int(duration), 60)
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

axes[0].plot(
    np.linspace(0, show_sec, show_sec * SR),
    audio[: show_sec * SR],
    linewidth=0.4,
    alpha=0.8,
)
axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude", xlim=(0, show_sec))

D = librosa.amplitude_to_db(np.abs(librosa.stft(audio[: show_sec * SR])), ref=np.max)
librosa.display.specshow(D, sr=SR, x_axis="time", y_axis="hz",
                         ax=axes[1], fmax=8000, cmap="magma")
axes[1].set_title(f"Spectrogram (first {show_sec}s)")
plt.tight_layout()
plt.show()

print(f"\\n▶ Listen (first {show_sec}s):")
display(Audio(audio[: show_sec * SR], rate=SR))
"""),

# ── Q1 ─────────────────────────────────────────────────────────────────────
md("""\
---
### ✏️ Q1 — Why Did You Choose This Audio?

1. What instrument or sound source did you upload?
2. Why do you think it is suitable for DDSP training? \
Which of the "works well" properties does it satisfy?
3. Look at the spectrogram above. \
Do you see clear horizontal harmonic bands? \
Describe the visual structure you observe.

---
**Your answer:**

> *1.*

> *2.*

> *3.*
"""),

# ── Part 2 header ──────────────────────────────────────────────────────────
md("""\
---
## Part 2: Preprocessing — Extracting Pitch & Loudness

DDSP does not train on raw audio. The model receives only two extracted signals:

- **Pitch (F0):** the fundamental frequency at each 10 ms frame (zero when no pitch is detected)
- **Loudness:** A-weighted RMS energy in dB at each frame

The next cell uses **torchcrepe** — a GPU-accelerated PyTorch port of CREPE, a neural \
pitch estimator. CREPE also returns a per-frame **voicing confidence** in `[0, 1]`. \
Frames whose confidence is below `0.5` are treated as **unvoiced** and set to `f0 = 0`. \
If torchcrepe fails or returns too few voiced frames (e.g., your audio is outside its \
training distribution), the code automatically falls back to librosa's PYIN.

On a T4 GPU this takes well under a minute even for long audio.

**What to check in the pitch plots below:**
- Pitch should be non-zero during sustained notes, zero in silence and breaths
- Contours should be smooth within a note and jump cleanly between notes
- If pitch is mostly zero, your audio may be too noisy, polyphonic, or outside \
the extractor's pitch range (50–2000 Hz)
"""),

# ── 6. Preprocess ──────────────────────────────────────────────────────────
code("""\
# @title 6. Extract Pitch & Loudness
# @markdown Pitch is estimated by torchcrepe (GPU). Frames with confidence < 0.5 are
# @markdown treated as unvoiced and set to f0 = 0.
import sys
sys.path.insert(0, ".")
import numpy as np
from ddsp.dataset import preprocess_audio, save_preprocessed

PREPROCESS_DIR = "./preprocessed/train"
BLOCK_SIZE = 160    # hop size → 100 frames per second
SIGNAL_LEN = 64000  # 4 seconds per training clip

print("Extracting pitch and loudness...")
signals, pitches, loudnesses = preprocess_audio(audio, SR, BLOCK_SIZE, SIGNAL_LEN)
save_preprocessed(PREPROCESS_DIR, signals, pitches, loudnesses)

print(f"\\n✓ {signals.shape[0]} clips saved to {PREPROCESS_DIR}/")
print(f"  signals:   {signals.shape}")
print(f"  pitches:   {pitches.shape}")
print(f"  loudness:  {loudnesses.shape}")
"""),

# ── 7. Visualize features ─────────────────────────────────────────────────
code("""\
# @title 7. Visualize Extracted Features
import matplotlib.pyplot as plt
import numpy as np

n_show = min(4, signals.shape[0])
fig, axes = plt.subplots(n_show, 2, figsize=(14, 2.8 * n_show))
if n_show == 1:
    axes = axes[np.newaxis, :]

for i in range(n_show):
    t = np.arange(pitches.shape[1]) / 100.0  # seconds at 100 Hz

    # Pitch: mask zeros so they appear as gaps
    p = pitches[i].copy().astype(float)
    p[p == 0] = np.nan
    axes[i, 0].plot(t, p, lw=0.8, color="steelblue")
    axes[i, 0].set(title=f"Clip {i + 1}: Pitch (F0)",
                   ylabel="Hz", xlabel="Time (s)", ylim=(0, 2000))

    axes[i, 1].plot(t, loudnesses[i], lw=0.8, color="coral")
    axes[i, 1].set(title=f"Clip {i + 1}: Loudness",
                   ylabel="dB", xlabel="Time (s)")

plt.suptitle("Extracted Features — First Clips", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Summary statistics
voiced = pitches[pitches > 0]
voiced_pct = 100.0 * voiced.size / pitches.size
print(f"Voiced frames : {voiced_pct:.1f}%")
if voiced.size:
    print(f"Pitch range   : {voiced.min():.0f} – {voiced.max():.0f} Hz "
          f"  (median {np.median(voiced):.0f} Hz)")
"""),

# ── Q2 ─────────────────────────────────────────────────────────────────────
md("""\
---
### ✏️ Q2 — Pitch Extraction Quality

1. What percentage of frames have detected pitch (voiced frames)? \
Does this match what you would expect given your audio?
2. Do the pitch contours look smooth and musically sensible?
3. If you see problems (many gaps, erratic jumps), what characteristics \
of your audio might be causing them?

---
**Your answer:**

> *1.*

> *2.*

> *3.*
"""),

# ── Part 3 header ──────────────────────────────────────────────────────────
md("""\
---
## Part 3: Training

The model will now learn to reconstruct your audio from just pitch and loudness. \
The harmonic synthesizer learns which overtones to emphasize; the noise synthesizer \
learns the inharmonic texture. Together they capture the characteristic timbre of your instrument.

**Training loss:** Multi-scale spectral loss — the L1 distance between the STFT magnitudes \
of the predicted and target audio, computed at 6 different window sizes simultaneously.

### Estimated training time on a T4 GPU

| Steps | Time | Expected quality |
|-------|------|-----------------|
| 10,000 | ~10 min | Basic timbre recognizable |
| 30,000 | ~30 min | Clearly sounds like the instrument |
| 50,000 | ~50 min | High quality — recommended |

A preview audio file is saved every 5,000 steps so you can hear the model improve.

> **Tip:** If your Colab session may time out, mount Google Drive (Cell 3) \
so the checkpoint survives a session reset.
"""),

# ── 8. Train ───────────────────────────────────────────────────────────────
code("""\
# @title 8. Train Your DDSP Model
# @markdown Set the number of steps and a name for this run, then run the cell.
# @markdown ⏱ 50,000 steps takes approximately 50 minutes on a T4 GPU.
import subprocess, sys, yaml, os

STEPS    = 50000         # @param {type:"integer"}
RUN_NAME = "my_instrument"  # @param {type:"string"}

# Write a config for this run
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["out_dir"] = PREPROCESS_DIR
with open("configs/run.yaml", "w") as f:
    yaml.safe_dump(cfg, f)

run_out = globals().get("DRIVE_SAVE_DIR") or "runs"
CHECKPOINT_PATH = os.path.join(run_out, RUN_NAME, "checkpoint.pt")

print(f"Training for {STEPS:,} steps")
print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}")
print("─" * 60)

result = subprocess.run([
    sys.executable, "train.py",
    "--config", "configs/run.yaml",
    "--name",   RUN_NAME,
    "--out",    run_out,
    "--steps",  str(STEPS),
])
if result.returncode == 0:
    print("\\n✓ Training complete!")
else:
    print(f"\\n⚠ Training exited with code {result.returncode}")
"""),

# ── 9. Training previews ──────────────────────────────────────────────────
code("""\
# @title 9. Listen to Training Progress
# @markdown Preview files are saved every 5,000 steps.
# @markdown You should hear the model get progressively better.
import glob, os
from IPython.display import Audio, display
import soundfile as sf

run_dir = os.path.join(globals().get("DRIVE_SAVE_DIR") or "runs", RUN_NAME)
previews = sorted(glob.glob(os.path.join(run_dir, "preview_*.wav")))

if not previews:
    print(f"No preview files found in {run_dir}/")
    print("Check that training completed and that RUN_NAME matches what you set in Cell 8.")
else:
    print(f"Found {len(previews)} preview(s):\\n")
    for path in previews:
        step_str = os.path.basename(path).replace("preview_", "").replace(".wav", "")
        step = int(step_str)
        audio_preview, sr_preview = sf.read(path)
        print(f"▶ Step {step:,}:")
        display(Audio(audio_preview, rate=sr_preview))
"""),

# ── Q3 ─────────────────────────────────────────────────────────────────────
md("""\
---
### ✏️ Q3 — Training Progress

1. How does the sound change between the earliest and latest preview?
2. At roughly which step did the model first start to sound recognizable as your instrument?
3. Did improvement continue steadily, or did it plateau at some point? \
If it plateaued, at around which step?

---
**Your answer:**

> *1.*

> *2.*

> *3.*
"""),

# ── Part 4 header ──────────────────────────────────────────────────────────
md("""\
---
## Part 4: Timbre Transfer

Your trained model now acts as a *timbre processor*. Upload any audio: the model will extract \
its pitch and loudness, then resynthesize it using the timbre it learned from your training audio.

The output keeps the **melody, rhythm, and dynamics** of the source but replaces \
the instrument sound entirely.

### Tips for choosing source audio
- A melody on a contrasting instrument creates the most striking demonstration
- Humming or whistling a melody works very well (clear pitch, easy to upload)
- A scale or arpeggio is useful for systematic comparisons
- Avoid chord-heavy or percussive material — the model can only follow one pitch at a time
"""),

# ── 10. Upload source ─────────────────────────────────────────────────────
code("""\
# @title 10. Upload Source Audio for Timbre Transfer
# @markdown This is the audio whose melody will be kept — timbre will change.
# @markdown A short clip of 10–60 seconds is enough for a first test.
from google.colab import files
import soundfile as sf
from IPython.display import Audio, display

print("Upload your source audio:")
uploaded_src = files.upload()
SOURCE_FILE = list(uploaded_src.keys())[0]

src_audio, src_sr = sf.read(SOURCE_FILE)
print(f"\\n▶ Your source audio ({SOURCE_FILE}):")
display(Audio(src_audio, rate=src_sr))
"""),

# ── 11. Run transfer ──────────────────────────────────────────────────────
code("""\
# @title 11. Run Timbre Transfer
import sys, os
sys.path.insert(0, ".")
from inference import timbre_transfer
import soundfile as sf
from IPython.display import Audio, display

if not os.path.exists(CHECKPOINT_PATH):
    print(f"⚠  Checkpoint not found at: {CHECKPOINT_PATH}")
    print("Make sure training completed successfully (Cell 8) and RUN_NAME matches.")
else:
    OUTPUT_FILE = "output_transfer.wav"
    timbre_transfer(SOURCE_FILE, CHECKPOINT_PATH, OUTPUT_FILE)

    src, ssr = sf.read(SOURCE_FILE)
    out, osr = sf.read(OUTPUT_FILE)

    print("▶ Source (original):")
    display(Audio(src, rate=ssr))
    print("▶ After timbre transfer:")
    display(Audio(out, rate=osr))
"""),

# ── 12. Download result ───────────────────────────────────────────────────
code("""\
# @title 12. Download Result
from google.colab import files
files.download(OUTPUT_FILE)
"""),

# ── Q4 ─────────────────────────────────────────────────────────────────────
md("""\
---
### ✏️ Q4 — Timbre Transfer Results

1. What source audio did you use, and why did you choose it?
2. What was successfully transferred from the source (pitch contour, rhythm, phrasing, dynamics)?
3. How would you describe the character of the output? \
In what ways does it sound different from both the source and the training audio?
4. Were there artifacts or failure modes? Where and why do you think they occurred?

---
**Your answer:**

> *1.*

> *2.*

> *3.*

> *4.*
"""),

# ── 13. Pitch shift ───────────────────────────────────────────────────────
code("""\
# @title 13. Pitch Shift Experiments
# @markdown Transposes the extracted F0 by a fixed number of semitones before synthesis.
# @markdown The model synthesizes the same melody at a different pitch.
import sys
sys.path.insert(0, ".")
from inference import timbre_transfer
import soundfile as sf
from IPython.display import Audio, display

SHIFTS = [-12, -7, -5, 0, 5, 7, 12]
LABELS = {
    -12: "− 1 octave",
     -7: "− perfect 5th",
     -5: "− perfect 4th",
      0: "  no shift",
      5: "+ perfect 4th",
      7: "+ perfect 5th",
     12: "+ 1 octave",
}

for shift in SHIFTS:
    out_file = f"output_shift_{shift:+d}.wav"
    timbre_transfer(SOURCE_FILE, CHECKPOINT_PATH, out_file,
                    pitch_shift_semitones=float(shift))
    audio_s, sr_s = sf.read(out_file)
    print(f"▶ {shift:+3d} semitones  ({LABELS[shift]}):")
    display(Audio(audio_s, rate=sr_s))
"""),

# ── Q5 ─────────────────────────────────────────────────────────────────────
md("""\
---
### ✏️ Q5 — Pitch Shifting

1. Which pitch shift sounded most natural? Which was most interesting or surprising?
2. What happened at extreme shifts (±12 semitones)? \
Did the model handle them well, or were there new artifacts?
3. Did the timbre quality change noticeably at different pitch ranges? \
Why might a model trained on, say, a violin in the range 200–1000 Hz \
struggle with very high or very low pitches?

---
**Your answer:**

> *1.*

> *2.*

> *3.*
"""),

# ── Part 5 header ──────────────────────────────────────────────────────────
md("""\
---
## Part 5: Creative Exploration

Now experiment freely. Run Cell 14 as many times as you like with different settings. \
Some directions to try:

- **Unexpected source:** speech, birdsong, environmental sound, another instrument family
- **Loudness offset:** `+10 dB` increases the perceived amplitude envelope; `-10 dB` softens it
- **Large pitch shifts:** try ±24 semitones — two full octaves
- **Chained transfer:** save an output from Cell 11, then upload it here as the source
- **Very short or long source:** how does clip duration affect the output?
"""),

# ── 14. Creative experiment ───────────────────────────────────────────────
code("""\
# @title 14. Creative Experiment
# @markdown Upload any source audio and adjust the parameters below.
# @markdown Run this cell multiple times with different settings.
from google.colab import files
import sys, os
sys.path.insert(0, ".")
from inference import timbre_transfer
import soundfile as sf
from IPython.display import Audio, display

PITCH_SHIFT     =  0.0  # @param {type:"slider", min:-24, max:24, step:0.5}
LOUDNESS_OFFSET =  0.0  # @param {type:"slider", min:-20, max:20, step:1}
EXPERIMENT_NAME = "experiment_1"  # @param {type:"string"}

print("Upload your creative source audio:")
src_uploaded = files.upload()

if src_uploaded:
    creative_source = list(src_uploaded.keys())[0]
    creative_output = f"{EXPERIMENT_NAME}.wav"

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠  Checkpoint not found: {CHECKPOINT_PATH}")
    else:
        timbre_transfer(
            source_path=creative_source,
            checkpoint_path=CHECKPOINT_PATH,
            output_path=creative_output,
            pitch_shift_semitones=PITCH_SHIFT,
            loudness_db_offset=LOUDNESS_OFFSET,
        )
        src, ssr = sf.read(creative_source)
        out, osr = sf.read(creative_output)
        print("\\n▶ Source:")
        display(Audio(src, rate=ssr))
        print("▶ Result:")
        display(Audio(out, rate=osr))
        files.download(creative_output)
"""),

# ── Q6 ─────────────────────────────────────────────────────────────────────
md("""\
---
### ✏️ Q6 — Creative Experiment

1. What source audio did you use, and what made you curious about this combination?
2. What settings did you use (pitch shift, loudness offset)?
3. Describe the result. What was surprising or unexpected?

---
**Your answer:**

> *1.*

> *2.*

> *3.*
"""),

# ── Final reflection ───────────────────────────────────────────────────────
md("""\
---
## Final Reflection

These questions ask you to think beyond the specific audio you worked with today.
"""),

md("""\
### ✏️ Q7 — What Would You Train Next?

If you could train DDSP on *any* sound in the world — any instrument, voice, animal, \
environment, machine — what would you choose?
What would you use as the source audio for timbre transfer, and why would this \
combination produce something interesting?

---
**Your answer:**

> *Type here*
"""),

md("""\
### ✏️ Q8 — What Is Lost?

DDSP uses only **pitch** and **loudness** as control signals. Everything else — vibrato \
speed and depth, spectral flux, articulation nuance, breath noise texture, \
micro-timing — must be either reconstructed by the model from these two signals or is \
permanently discarded.

Based on your listening experience in this assignment: \
what information do you think is *lost* in this simplification? \
How does that loss show up concretely in the sounds you heard?

---
**Your answer:**

> *Type here*
"""),

]  # end cells list


# ---------------------------------------------------------------------------
# Assemble and write notebook
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {
            "name": "DDSP Assignment: Build Your Own Timbre Model",
            "provenance": [],
        },
    },
    "cells": cells,
}

out_path = pathlib.Path("notebooks/assignment.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Written: {out_path}")
