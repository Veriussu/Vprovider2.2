"""One-shot Qwen3-TTS worker. The process exits after writing one WAV file."""

import json
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
import sys

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    request_path, output_path = sys.argv[1], sys.argv[2]
    with open(request_path, encoding="utf-8") as handle:
        req = json.load(handle)
    req["text"] = normalize_tts_text(req.get("text", ""))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen3TTSModel.from_pretrained(
        req["model_path"],
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if req["mode"] == "clone":
        wavs, sample_rate = model.generate_voice_clone(
            text=req["text"],
            language=req.get("language") or "Auto",
            ref_audio=req["reference_path"],
            ref_text=req.get("reference_text") or None,
            x_vector_only_mode=not bool(req.get("reference_text")),
        )
    else:
        wavs, sample_rate = model.generate_custom_voice(
            text=req["text"],
            language=req.get("language") or "Auto",
            speaker=req.get("speaker") or "Ryan",
            instruct=req.get("instruct") or None,
        )
    sf.write(output_path, wavs[0], sample_rate)


if __name__ == "__main__":
    main()
