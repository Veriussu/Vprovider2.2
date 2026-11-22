#!/usr/bin/env bash
# XTTS-v2 eklentisi için ayrı çalışma ortamı.
# Not: Daha önce XTTS coqui paketi paylaşımlı runtime/qwen-tts/venv içindeydi;
# artık kendi venv'inde kurulur ki 'Kaldır' işlemi qwen-tts venv'ini silmesin.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/runtime/plugins/xtts_v2/venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  mkdir -p "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip
# Coqui XTTS-v2 + torchcodec (PyTorch 2.9+ ses IO için gerekli)
"$VENV_DIR/bin/pip" install "coqui-tts[codec]"

echo "XTTS-v2 çalışma ortamı hazır: $VENV_DIR"
