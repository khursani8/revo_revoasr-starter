# Starter Submission: Whisper Tiny

A ready-to-upload baseline using OpenAI's Whisper tiny model (~75 MB).

## How It Works

The evaluation container has **no internet**. You must bundle model weights
inside the submission ZIP. This example provides a `build.sh` script that
downloads the model on **your machine** (which has internet) and packages
everything into `submission.zip`.

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/khursanirevo/revoasr-starter.git
cd revoasr-starter

# 2. Install dependencies (on YOUR machine)
pip install transformers torch soundfile

# 3. Download model and create submission.zip
bash build.sh

# 4. Upload the resulting submission.zip to the competition platform
```

## Submission Structure

The generated `submission.zip` contains:

```
├── inference.py     # Runs inside the container (offline, CPU only)
└── model/           # Whisper tiny weights (downloaded by build.sh)
```

## Container Environment

The base image provides these pre-installed (you don't need to bundle them):

- Python 3.11
- PyTorch (CPU only)
- transformers, soundfile, librosa, scipy, jiwer, numpy

What you **must** bundle:
- `inference.py` (required)
- Model weights (anything your inference.py needs to load)
- `wheels/*.whl` (optional, if you need extra Python packages)

## Customization Ideas

- **Better accuracy**: Change `MODEL_NAME` in build.sh to `openai/whisper-base` or `openai/whisper-small`
- **Faster inference**: Use `faster-whisper` (CTranslate2) — include the wheel in `wheels/`
- **Beam search**: Change `model.generate(inputs)` to `model.generate(inputs, num_beams=5)`
- **Language hint**: Add `language="en"` to processor call if you know the language
- **Post-processing**: Add text normalization, punctuation restoration, etc.

## Expected Performance

Whisper tiny on a single CPU core:
- RTF: ~0.03–0.10 (well under the 0.08 limit)
- WER: varies by language and domain
