#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    MusicGen Servis Adaptörü
#  Dosya:    plugins/adapters/musicgen_server.py
#  Amaç:     facebook/musicgen-small modelini (transformers) bir yerel HTTP
#            servisi olarak sunar. Model tembel yüklenir; servis ayakta
#            kalır ama boşta GPU'yu bırakır (bkz. _runtime_model).
#  Uçlar:    GET  /health    -> {"ok","loaded","sample_rate"}
#            POST /generate  -> {"prompt","duration"} -> {"audio": <b64 wav>}
#  Çalışma:  Eklentinin venv'iyle: --port --weights [--idle-timeout]
# ─────────────────────────────────────────────────────────────

import argparse
import base64
import io
import json
import os
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _runtime_model import LazyGpuModel, free_torch_model

warnings.filterwarnings("ignore")

_HANDLE: LazyGpuModel
_WEIGHTS = ""
_SAMPLE_RATE = 32000


def _build():
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    print(f"[musicgen] model yükleniyor: {_WEIGHTS}", flush=True)
    processor = AutoProcessor.from_pretrained(_WEIGHTS)
    model = MusicgenForConditionalGeneration.from_pretrained(_WEIGHTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return {"processor": processor, "model": model}


def _free(bundle) -> None:
    old = bundle.pop("model", None)
    bundle.pop("processor", None)
    if old is not None:
        free_torch_model(old)


def _generate(bundle, prompt: str, duration: float) -> tuple[bytes, int]:
    import numpy as np
    import torch
    import scipy.io.wavfile as wavfile

    processor = bundle["processor"]
    model = bundle["model"]
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(model.device)
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    # MusicGen: saniyede ~50 ses token'ı üretir.
    max_new_tokens = max(32, int(duration * 50))
    with torch.no_grad():
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=max_new_tokens,
        )
    data = audio[0, 0].cpu().numpy()
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    return buf.getvalue(), sample_rate


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
            self._json(200, {"ok": True, "loaded": _HANDLE.loaded, "sample_rate": _SAMPLE_RATE})
        else:
            self._json(404, {"error": "bulunamadı"})

    def do_POST(self):  # noqa: N802
        if self.path != "/generate":
            self._json(404, {"error": "bulunamadı"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            duration = float(request.get("duration", 10.0))
            with _HANDLE.use() as bundle:
                audio, sample_rate = _generate(bundle, request.get("prompt", ""), duration)
            self._json(200, {
                "audio": base64.b64encode(audio).decode("ascii"),
                "sample_rate": sample_rate,
            })
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._json(500, {"error": str(exc)})


def main() -> None:
    global _HANDLE, _WEIGHTS, _SAMPLE_RATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--src", default="")
    parser.add_argument("--idle-timeout", type=float,
                        default=float(os.environ.get("VPROVIDER_PLUGIN_IDLE_SECONDS", "120")))
    args = parser.parse_args()

    _WEIGHTS = args.weights
    _HANDLE = LazyGpuModel(_build, _free, idle_seconds=args.idle_timeout, name="musicgen")
    _HANDLE.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    idle_note = "kapalı" if args.idle_timeout < 0 else f"{args.idle_timeout:g}s"
    print(f"[musicgen] servis 127.0.0.1:{args.port} dinliyor (boşta bırakma: {idle_note}).", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
