#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    SeedVR2 Servis Adaptörü
#  Dosya:    plugins/adapters/seedvr2_server.py
#  Amaç:     ByteDance-Seed/SeedVR2-3B video onarım modelini sarar.
#            Servis başlat/durdur/log yaşam döngüsü ve sağlık denetimi
#            çalışır; çıkarım çok ağır olduğundan (≥80 GB VRAM) bu uç
#            nokta ortam yetersizse açık bir hata döner.
#  Uçlar:    GET  /health   -> {"ok","loaded","src_ok","weights_ok"}
#            POST /restore  -> {"video": <b64>} (yeterli donanım gerekir)
#  Çalışma:  Eklentinin venv'iyle: --port --weights --src
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

_SRC = ""
_WEIGHTS = ""
_PYTHON = sys.executable


def _find_inference() -> str:
    """Motora ait çıkarım betiğini bulur."""
    candidates = (
        "projects/video_diffusion_sr/inference.py",
        "projects/video_diffusion_sr/seedvr2/inference.py",
        "inference.py",
    )
    for rel in candidates:
        path = os.path.join(_SRC, rel)
        if os.path.isfile(path):
            return path
    return ""


def _restore(video_path: str, out_dir: str) -> str:
    script = _find_inference()
    if not script:
        raise RuntimeError(
            "SeedVR2 çıkarım betiği bulunamadı; kaynak kod eksik olabilir."
        )
    out_path = os.path.join(out_dir, "restored.mp4")
    cmd = [
        _PYTHON, script,
        "--video_path", video_path,
        "--output_path", out_path,
    ]
    print("[seedvr2] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=_SRC, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Çıkarım başarısız (kod {proc.returncode})")
    if os.path.isfile(out_path):
        return out_path
    raise RuntimeError("Çıktı videosu bulunamadı.")


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
                "loaded": bool(_find_inference()) and os.path.isdir(_WEIGHTS),
                "src_ok": os.path.isdir(_SRC),
                "weights_ok": os.path.isdir(_WEIGHTS),
            })
        else:
            self._json(404, {"error": "bulunamadı"})

    def do_POST(self):  # noqa: N802
        if self.path != "/restore":
            self._json(404, {"error": "bulunamadı"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            with tempfile.TemporaryDirectory(prefix="vprovider-seedvr2-") as out_dir:
                video_path = os.path.join(out_dir, "in.mp4")
                with open(video_path, "wb") as handle:
                    handle.write(base64.b64decode(request.get("video") or request.get("image") or ""))
                result_path = _restore(video_path, out_dir)
                with open(result_path, "rb") as handle:
                    out_b64 = base64.b64encode(handle.read()).decode("ascii")
            self._json(200, {"video": out_b64})
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._json(500, {"error": str(exc)})


def main() -> None:
    global _SRC, _WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--src", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--idle-timeout", type=float, default=120.0)
    args = parser.parse_args()

    _SRC = args.src
    _WEIGHTS = args.weights
    print(f"[seedvr2] src={_SRC} weights={_WEIGHTS}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[seedvr2] servis 127.0.0.1:{args.port} dinliyor.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
