#!/usr/bin/env bash
# Build submission.zip: downloads model weights and packages everything.
#
# Run this on YOUR machine (needs internet). The resulting submission.zip
# is self-contained — it works inside the offline evaluation container.
#
# Usage:
#   cd examples/starter-whisper
#   bash build.sh
#
# Requirements (on your machine):
#   pip install transformers torch soundfile

set -e

MODEL_NAME="openai/whisper-tiny"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model"
ZIP_PATH="$SCRIPT_DIR/submission.zip"

# Download model weights using Python
echo "Downloading $MODEL_NAME ..."
python -c "
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import shutil, os
model_dir = '$MODEL_DIR'
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)
processor = WhisperProcessor.from_pretrained('$MODEL_NAME')
model = WhisperForConditionalGeneration.from_pretrained('$MODEL_NAME')
processor.save_pretrained(model_dir)
model.save_pretrained(model_dir)
print('Saved model to', model_dir)
"

# Create submission.zip with inference.py + model/
echo "Creating submission.zip ..."
cd "$SCRIPT_DIR"
rm -f "$ZIP_PATH"
zip -r "$ZIP_PATH" inference.py model/

SIZE=$(du -h "$ZIP_PATH" | cut -f1)
echo ""
echo "Done! Created submission.zip ($SIZE)"
echo "Upload it via the competition platform:"
echo "  curl -X POST http://<host>/api/submissions/upload -H 'X-API-Key: YOUR_KEY' -F 'file=@submission.zip'"
