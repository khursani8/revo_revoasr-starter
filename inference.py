"""Starter submission using Whisper tiny via HuggingFace Transformers.

This is a baseline you can improve upon. It loads a bundled whisper-tiny model
and transcribes every .wav file in the test set.

To customize:
  - Swap the model (e.g. whisper-small, whisper-base) in build.sh and re-run it
  - Add your own preprocessing / postprocessing
  - Try beam search or temperature fallback for better accuracy

Environment variables (set by the container):
  TEST_DIR   - path to test audio files (e.g. /data/test)
  OUTPUT_DIR - path to write predictions.json (e.g. /data/output)

NOTE: The container has NO internet access. Model weights must be bundled
      in the submission ZIP alongside this file.
"""

import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Paths
TEST_DIR = os.environ.get("TEST_DIR", "/data/test")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")

# Model weights are bundled alongside this script in the "model/" directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# Target sample rate for Whisper
TARGET_SR = 16000


def _load_audio(path: str) -> np.ndarray:
    """Load audio file and resample to 16 kHz mono float32."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly

        gcd = np.gcd(TARGET_SR, sr)
        audio = resample_poly(audio, TARGET_SR // gcd, sr // gcd)
    return audio


def main() -> None:
    device = torch.device("cpu")

    print(f"Loading model from {MODEL_DIR} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    print("Model loaded (CPU).")

    wav_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".wav"))
    print(f"Found {len(wav_files)} audio files.")

    predictions = []

    for i, wav_file in enumerate(wav_files):
        file_id = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(TEST_DIR, wav_file)

        start = time.perf_counter()

        audio = _load_audio(wav_path)
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inputs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        elapsed = time.perf_counter() - start

        predictions.append({
            "file_id": file_id,
            "text": text,
            "inference_time": round(elapsed, 4),
        })

        if (i + 1) % 100 == 0 or i == len(wav_files) - 1:
            print(f"  {i + 1}/{len(wav_files)} done")

    # Write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    total_time = sum(p["inference_time"] for p in predictions)
    print(f"Wrote {len(predictions)} predictions to {out_path}")
    print(f"Total inference time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
