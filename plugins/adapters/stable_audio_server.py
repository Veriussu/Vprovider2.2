#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Stable Audio Open Servis Adaptörü
#  Dosya:    plugins/adapters/stable_audio_server.py
#  Amaç:     stabilityai/stable-audio-open-1.0 modelini (stable-audio-tools)
#            yerel HTTP servisi olarak sunar. Model tembel yüklenir; servis
#            ayakta kalır ama boşta GPU'yu bırakır (bkz. _runtime_model).
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
_SAMPLE_RATE = 44100


def _build():
    import torch
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    print(f"[stable-audio] model yükleniyor: {_WEIGHTS}", flush=True)
    with open(os.path.join(_WEIGHTS, "model_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    model = create_model_from_config(config)
    ckpt = os.path.join(_WEIGHTS, "model.safetensors")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(_WEIGHTS, "model.ckpt")
    model.load_state_dict(load_ckpt_state_dict(ckpt))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return {"model": model, "config": config}


def _free(bundle) -> None:
    old = bundle.pop("model", None)
    bundle.pop("config", None)
    if old is not None:
        free_torch_model(old)


def _generate(bundle, prompt: str, duration: float) -> tuple[bytes, int]:
    import torch
    import torchaudio
    from einops import rearrange
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model = bundle["model"]
    config = bundle["config"]
    device = next(model.parameters()).device
    sample_rate = int(config["sample_rate"])
    sample_size = int(config["sample_size"])
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": float(duration),
    }]
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    buf = io.BytesIO()
    torchaudio.save(buf, output, sample_rate, format="wav")
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
    _HANDLE = LazyGpuModel(_build, _free, idle_seconds=args.idle_timeout, name="stable-audio")
    _HANDLE.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    idle_note = "kapalı" if args.idle_timeout < 0 else f"{args.idle_timeout:g}s"
    print(f"[stable-audio] servis 127.0.0.1:{args.port} dinliyor (boşta bırakma: {idle_note}).", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
