"""Local Qwen3-TTS discovery and one-shot inference adapter."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from app.config import settings
from app.tts_text import normalize_tts_text
from app.runtime_coord import llm_execution_lock
from app.voice_references import get as get_reference

# Coqui XTTS-v2 varsayılan konuşmacıları (CustomVoice modelleri için)
DEFAULT_SPEAKERS = [
    "Alex", "Amy", "Anya", "Bjorn", "Daniel", "Dora", "Elliot", "Freya",
    "Isabella", "Jason", "Kendra", "Liam", "Mila", "Noah", "Olivia",
    "Sari", "Theodore", "Thor", "Tim", "Winnie",
]
# Qwen3-TTS hazır konuşmacıları
QWEN_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


class LocalTTSError(RuntimeError):
    pass


def _model_info(path: Path) -> dict | None:
    config_path = path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Qwen3-TTS formatı
    if config.get("model_type") == "qwen3_tts":
        model_type = str(config.get("tts_model_type") or "").lower()
        mode = "clone" if model_type == "base" else "custom_voice"
        return {
            "id": path.name,
            "path": str(path),
            "engine": "qwen3-tts",
            "mode": mode,
            "supports_clone": mode == "clone",
            "speakers": [] if mode == "clone" else QWEN_SPEAKERS,
            "languages": LANGUAGES,
        }
    # Coqui XTTS formatı (model: "xtts")
    if str(config.get("model") or "").lower() == "xtts":
        lang = [str(l).lower() for l in config.get("languages", []) or []]
        if not lang:
            lang = ["en", "tr", "es", "fr", "de", "it", "pt", "ru", "zh-cn", "ja"]
        return {
            "id": path.name,
            "path": str(path),
            "engine": "coqui-tts",
            "mode": "clone",
            "supports_clone": True,
            "speakers": [],  # XTTS klon tabanlıdır, konuşmacı listesi yoktur
            "languages": ["Auto"] + lang,
        }
    return None


def list_models() -> list[dict]:
    base = settings.models_path / "tts"
    if not base.is_dir():
        return []
    return [info for path in sorted(base.iterdir()) if path.is_dir() if (info := _model_info(path))]


def get_model(model_id: str) -> dict | None:
    return next((model for model in list_models() if model["id"] == model_id), None)


async def synthesize(
    model_id: str,
    text: str,
    language: str = "Auto",
    speaker: str = "Ryan",
    instruct: str = "",
    reference_voice_id: str = "",
    reference_text: str = "",
) -> tuple[bytes, str]:
    text = normalize_tts_text(text)
    model = get_model(model_id)
    if model is None:
        raise LocalTTSError("Seçilen yerel TTS modeli bulunamadı.")
    if model["supports_clone"] and not reference_voice_id:
        raise LocalTTSError("Bu ses klonlama modeli için referans ses seçmelisiniz.")
    if not model["supports_clone"] and reference_voice_id:
        raise LocalTTSError("Bu CustomVoice modeli referans ses klonlamaz; Base varyantını kullanın.")

    reference_path = ""
    if reference_voice_id:
        reference = get_reference(reference_voice_id)
        if reference is None:
            raise LocalTTSError("Referans ses bulunamadı.")
        reference_path = str(reference[1].resolve())

    runner = Path("runtime/qwen-tts/venv/bin/python")
    if not runner.is_file():
        raise LocalTTSError("Yerel TTS çalışma ortamı kurulu değil (vprovider update ile kurulur).")
    if model["engine"] == "coqui-tts":
        worker = Path("scripts/coqui_tts_worker.py")
    else:
        worker = Path("scripts/qwen_tts_worker.py")

    request = {
        "model_path": str(Path(model["path"]).resolve()),
        "mode": "clone" if model["supports_clone"] else "custom_voice",
        "text": text,
        "language": language,
        "speaker": speaker,
        "instruct": instruct,
        "reference_path": reference_path,
        "reference_text": reference_text,
    }
    with tempfile.TemporaryDirectory(prefix="vprovider-tts-") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        output_path = Path(temp_dir) / "output.wav"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        async with llm_execution_lock:
            process = await asyncio.create_subprocess_exec(
                str(runner), str(worker), str(request_path), str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise LocalTTSError(message[-1] if message else "TTS üretimi başarısız.")
        if not output_path.is_file():
            raise LocalTTSError("TTS ses çıktısı oluşturmadı.")
        return output_path.read_bytes(), "audio/wav"
