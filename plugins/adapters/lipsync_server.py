#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Dudak Senkronu (Lipsync) Servis Adaptörü
#  Dosya:    plugins/adapters/lipsync_server.py
#  Amaç:     LatentSync / MuseTalk motorlarından birini kalıcı bir yerel
#            HTTP servisi olarak sarar. Model ağırlıkları motorun kendi
#            çıkarım (inference) betiği aracılığıyla kullanılır.
#  Uçlar:    GET  /health   -> {"ok", "loaded", "engine"}
#            POST /lipsync  -> {"image"/"video": <b64>, "audio": <b64>}
#                              yanıt: {"video": <b64 mp4>}
#  Çalışma:  Eklentinin venv python'u ile:
#            --port --engine latent_sync|muse_talk --weights --src
#  Not:      Motor çıkarım arayüzleri sürüme göre değişebilir; bu adaptör
#            bilinen CLI'leri çağırır ve oluşan hatayı log'a yazar.
# ─────────────────────────────────────────────────────────────

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ENGINE = ""
_SRC = ""
_WEIGHTS = ""
_PYTHON = sys.executable


def _run_engine(video_path: str, audio_path: str, out_dir: str) -> str:
    """Motorun çıkarım betiğini çalıştırır ve üretilen videonun yolunu döner."""
    if _ENGINE == "latent_sync":
        ckpt = os.path.join(_WEIGHTS, "checkpoints", "latentsync_unet.pt")
        out_path = os.path.join(out_dir, "out.mp4")
        cmd = [
            _PYTHON, "-m", "scripts.inference",
            "--inference_ckpt_path", ckpt,
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", out_path,
        ]
    elif _ENGINE == "muse_talk":
        config_path = os.path.join(out_dir, "inference.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(
                "task_0:\n"
                f"  video_path: {video_path}\n"
                f"  audio_path: {audio_path}\n"
                "  bbox_shift: 0\n"
            )
        cmd = [_PYTHON, "-m", "scripts.inference", "--inference_config", config_path]
        out_path = ""
    else:
        raise RuntimeError(f"Bilinmeyen motor: {_ENGINE}")

    print("[lipsync] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=_SRC, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Çıkarım başarısız (kod {proc.returncode})")

    if out_path and os.path.isfile(out_path):
        return out_path
    # MuseTalk: results/ altındaki en yeni mp4'ü bul
    candidates = []
    for root, _dirs, files in os.walk(_SRC):
        for name in files:
            if name.endswith(".mp4"):
                full = os.path.join(root, name)
                candidates.append((os.path.getmtime(full), full))
    if not candidates:
        raise RuntimeError("Çıktı videosu bulunamadı.")
    return max(candidates)[1]


def _to_file(data_b64: str, suffix: str, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name + suffix)
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(data_b64))
    return path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {
                "ok": True,
                "loaded": os.path.isdir(_SRC) and os.path.isdir(_WEIGHTS),
                "engine": _ENGINE,
            })
        else:
            self._json(404, {"error": "bulunamadı"})

    def do_POST(self):  # noqa: N802
        if self.path != "/lipsync":
            self._json(404, {"error": "bulunamadı"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            with tempfile.TemporaryDirectory(prefix="vprovider-lipsync-") as out_dir:
                video_b64 = request.get("video") or request.get("image")
                video_path = _to_file(video_b64, ".mp4", out_dir, "in")
                audio_path = _to_file(request.get("audio", ""), ".wav", out_dir, "in")
                result_path = _run_engine(video_path, audio_path, out_dir)
                with open(result_path, "rb") as handle:
                    out_b64 = base64.b64encode(handle.read()).decode("ascii")
            self._json(200, {"video": out_b64, "engine": _ENGINE})
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._json(500, {"error": str(exc)})


def main() -> None:
    global _ENGINE, _SRC, _WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--engine", default="latent_sync")
    parser.add_argument("--src", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--idle-timeout", type=float, default=120.0)
    args = parser.parse_args()

    _ENGINE = args.engine
    _SRC = args.src
    _WEIGHTS = args.weights
    print(f"[lipsync] motor={_ENGINE} src={_SRC} weights={_WEIGHTS}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[lipsync] servis 127.0.0.1:{args.port} dinliyor.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
