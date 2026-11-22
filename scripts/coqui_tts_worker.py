"""One-shot Coqui XTTS-v2 worker. The process exits after writing one WAV file."""

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
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import soundfile as sf
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def main() -> None:
    request_path, output_path = sys.argv[1], sys.argv[2]
    with open(request_path, encoding="utf-8") as handle:
        req = json.load(handle)

    model_path = req["model_path"]
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=model_path,
        eval=True,
        use_deepspeed=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text = normalize_tts_text(req.get("text", ""))
    reference_path = req.get("reference_path") or ""
    if not reference_path:
        raise SystemExit("XTTS klonlama için referans ses gerekli.")

    language = (req.get("language") or "en").lower()
    if language in ("auto", ""):
        language = "en"

    outputs = model.synthesize(
        text,
        config,
        speaker_wav=reference_path,
        language=language,
        enable_text_splitting=True,
    )
    wav = outputs["wav"]
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    sample_rate = int(config.audio.output_sample_rate)
    sf.write(output_path, wav, sample_rate)


if __name__ == "__main__":
    main()
