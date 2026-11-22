#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Real-ESRGAN Servis Adaptörü
#  Dosya:    plugins/adapters/esrgan_server.py
#  Amaç:     ai-forever/Real-ESRGAN modelini yerel HTTP servisi olarak
#            sunar. Model tembel yüklenir; servis ayakta kalır ama boşta
#            GPU'yu bırakır (bkz. _runtime_model).
#  Uçlar:    GET  /health    -> {"ok", "loaded", "scale"}
#            POST /upscale   -> {"image": <b64 png>}
#  Çalışma:  Eklentinin venv'iyle: --port --weights --scale [--idle-timeout]
# ─────────────────────────────────────────────────────────────

import argparse
import base64
import io
import json
import os
import sys
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _runtime_model import LazyGpuModel, free_torch_model

warnings.filterwarnings("ignore")

_HANDLE: LazyGpuModel
_SRC = ""
_WEIGHTS = "weights"
_SCALE = 4


def _build():
    if _SRC and os.path.isdir(_SRC) and _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import torch
    from RealESRGAN import RealESRGAN  # sberbank-ai/Real-ESRGAN paketi

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esrgan] cihaz={device} ölçek={_SCALE} yükleniyor...", flush=True)
    model = RealESRGAN(device, scale=_SCALE)
    os.makedirs(_WEIGHTS, exist_ok=True)
    weight_file = os.path.join(_WEIGHTS, f"RealESRGAN_x{_SCALE}.pth")
    # Ağırlık yoksa kütüphane HuggingFace'ten indirir.
    model.load_weights(weight_file, download=True)
    return {"model": model}


def _free(bundle) -> None:
    old = bundle.pop("model", None)
    if old is None:
        return
    # Sarmalayıcının iç torch modülünü CPU'ya taşımayı dene.
    inner = getattr(old, "model", None)
    if inner is not None and callable(getattr(inner, "to", None)):
        try:
            inner.to("cpu")
        except Exception:
            pass
    free_torch_model(old)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # erişim gürültüsünü sustur
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
            self._json(200, {"ok": True, "loaded": _HANDLE.loaded, "scale": _SCALE})
        else:
            self._json(404, {"error": "bulunamadı"})

    def do_POST(self):  # noqa: N802
        if self.path != "/upscale":
            self._json(404, {"error": "bulunamadı"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            from PIL import Image

            raw = base64.b64decode(request.get("image", ""))
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            with _HANDLE.use() as bundle:
                result = bundle["model"].predict(image)
            out = io.BytesIO()
            result.save(out, format="PNG")
            self._json(200, {"image": base64.b64encode(out.getvalue()).decode("ascii")})
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._json(500, {"error": str(exc)})


def main() -> None:
    global _HANDLE, _SRC, _WEIGHTS, _SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--src", default="")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--idle-timeout", type=float,
                        default=float(os.environ.get("VPROVIDER_PLUGIN_IDLE_SECONDS", "120")))
    args = parser.parse_args()

    _SRC = args.src
    _WEIGHTS = args.weights
    _SCALE = args.scale
    _HANDLE = LazyGpuModel(_build, _free, idle_seconds=args.idle_timeout, name="esrgan")
    _HANDLE.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    idle_note = "kapalı" if args.idle_timeout < 0 else f"{args.idle_timeout:g}s"
    print(f"[esrgan] servis 127.0.0.1:{args.port} dinliyor (boşta bırakma: {idle_note}).", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
