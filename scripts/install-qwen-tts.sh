#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/runtime/qwen-tts/venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install qwen-tts
# Coqui XTTS-v2 desteği (torchcodec, PyTorch 2.9+ ses IO için gerekli)
"$VENV_DIR/bin/pip" install "coqui-tts[codec]"

echo "Yerel TTS çalışma ortamı hazır: $VENV_DIR"
