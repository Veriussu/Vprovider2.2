#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Coqui XTTS-v2 Servis Adaptörü
#  Dosya:    plugins/adapters/tts_server.py
#  Amaç:     Coqui XTTS-v2 ses klonlama modelini yerel HTTP servisi olarak
#            sunar. Model tembel yüklenir; servis ayakta kalır ama boşta
#            GPU'yu bırakır (bkz. _runtime_model).
#  Uçlar:    GET  /health      -> {"ok", "loaded", "sample_rate"}
#            POST /synthesize  -> {"text","language","reference_path"|"reference_b64"}
#                                 yanıt: {"audio": <b64 wav>, "sample_rate": N}
#  Çalışma:  runtime/qwen-tts/venv python'u ile --port --weights [--idle-timeout]
# ─────────────────────────────────────────────────────────────

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from app.tts_text import normalize_tts_text
except Exception:
    def normalize_tts_text(text):
        return text
import tempfile
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _runtime_model import LazyGpuModel, free_torch_model

warnings.filterwarnings("ignore")

_HANDLE: LazyGpuModel
_WEIGHTS = ""
_SAMPLE_RATE = 24000


def _build():
    global _SAMPLE_RATE
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    print(f"[xtts] model yükleniyor: {_WEIGHTS}", flush=True)
    config = XttsConfig()
    config.load_json(os.path.join(_WEIGHTS, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=_WEIGHTS, eval=True, use_deepspeed=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    _SAMPLE_RATE = int(config.audio.output_sample_rate)
    return {"model": model, "config": config}


def _free(bundle) -> None:
    old = bundle.pop("model", None)
    bundle.pop("config", None)
    if old is not None:
        free_torch_model(old)


def _synthesize(bundle, text: str, language: str, reference_path: str) -> bytes:
    import soundfile as sf
    import torch

    model = bundle["model"]
    config = bundle["config"]
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=reference_path or None,
        language=(language or "en").lower(),
        enable_text_splitting=True,
    )
    wav = outputs["wav"]
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, wav, _SAMPLE_RATE, format="WAV")
    return buf.getvalue()


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
        if self.path != "/synthesize":
            self._json(404, {"error": "bulunamadı"})
            return
        temp_path = ""
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            reference_path = request.get("reference_path") or ""
            reference_b64 = request.get("reference_b64")
            if reference_b64 and not reference_path:
                handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                handle.write(base64.b64decode(reference_b64))
                handle.close()
                temp_path = reference_path = handle.name
            with _HANDLE.use() as bundle:
                audio = _synthesize(
                    bundle, normalize_tts_text(request.get("text", "")), request.get("language", "en"), reference_path
                )
            self._json(200, {
                "audio": base64.b64encode(audio).decode("ascii"),
                "sample_rate": _SAMPLE_RATE,
            })
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._json(500, {"error": str(exc)})
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


def main() -> None:
    global _HANDLE, _WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--src", default="")
    parser.add_argument("--idle-timeout", type=float,
                        default=float(os.environ.get("VPROVIDER_PLUGIN_IDLE_SECONDS", "120")))
    args = parser.parse_args()

    _WEIGHTS = args.weights
    _HANDLE = LazyGpuModel(_build, _free, idle_seconds=args.idle_timeout, name="xtts")
    _HANDLE.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    idle_note = "kapalı" if args.idle_timeout < 0 else f"{args.idle_timeout:g}s"
    print(f"[xtts] servis 127.0.0.1:{args.port} dinliyor (boşta bırakma: {idle_note}).", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
