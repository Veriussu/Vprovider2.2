#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Özel GGUF Runtime Kurulumu
#  Dosya:    scripts/install-prism-bonsai.sh
#  Amaç:     Prism/Bonsai PQ2_0/PTQ1_0 GGUF dosyaları için stock
#            llama.cpp'ten ayrı bir llama-server binary'si kurar.
#  Mekanik:  runtime/prism/src altına PrismML llama.cpp fork'u çekilir,
#            CUDA varsa CUDA ile, yoksa CPU ile build edilir. Çıktı
#            runtime/prism/bin/llama-server konumuna bağlanır/kopyalanır.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/runtime/prism"
SRC_DIR="${RUNTIME_DIR}/src"
BIN_DIR="${RUNTIME_DIR}/bin"
REPO_URL="${PRISM_LLAMA_REPO:-https://github.com/PrismML-Eng/llama.cpp.git}"

mkdir -p "${RUNTIME_DIR}" "${BIN_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "git bulunamadı; Prism runtime kurulamaz." >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake bulunamadı; Prism runtime kurulamaz." >&2
  exit 1
fi

if [ ! -d "${SRC_DIR}/.git" ]; then
  echo " • PrismML llama.cpp fork'u indiriliyor..."
  git clone --depth 1 "${REPO_URL}" "${SRC_DIR}"
else
  echo " • PrismML llama.cpp fork'u güncelleniyor..."
  git -C "${SRC_DIR}" pull --ff-only
fi

cmake_flags=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  cmake_flags+=("-DGGML_CUDA=ON")
fi

echo " • llama-server derleniyor (${cmake_flags[*]:-CPU})..."
cmake -S "${SRC_DIR}" -B "${SRC_DIR}/build" "${cmake_flags[@]}"
cmake --build "${SRC_DIR}/build" -j --target llama-server

server_bin="${SRC_DIR}/build/bin/llama-server"
if [ ! -x "${server_bin}" ]; then
  echo "llama-server çıktısı bulunamadı: ${server_bin}" >&2
  exit 1
fi

ln -sf "${server_bin}" "${BIN_DIR}/llama-server"
echo " ✓ Prism/Bonsai runtime hazır: ${BIN_DIR}/llama-server"
echo "   .env içinde PRISM_LLAMA_SERVER_BIN boşsa VProvider bu yolu otomatik kullanır."
