"""Persistent reference-audio storage for voice cloning engines."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from app.config import settings

MAX_VOICE_REFERENCE_BYTES = 30 * 1024 * 1024
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
}


def storage_dir() -> Path:
    path = Path(settings.data_dir) / "voice-references"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name).name).strip(" .")
    return cleaned[:120] or "reference-audio"


def _metadata_path(reference_id: str) -> Path:
    return storage_dir() / f"{reference_id}.json"


def save(filename: str, content_type: str, payload: bytes) -> dict:
    if not payload:
        raise ValueError("Ses dosyası boş olamaz.")
    if len(payload) > MAX_VOICE_REFERENCE_BYTES:
        raise ValueError("Ses dosyası en fazla 30 MB olabilir.")
    suffix = ALLOWED_AUDIO_TYPES.get(content_type)
    if suffix is None:
        suffix = Path(filename).suffix.lower()
        if suffix not in set(ALLOWED_AUDIO_TYPES.values()):
            raise ValueError("Desteklenmeyen ses biçimi. MP3, WAV, FLAC, OGG, M4A veya WEBM kullanın.")

    reference_id = uuid4().hex
    audio_path = storage_dir() / f"{reference_id}{suffix}"
    meta = {
        "id": reference_id,
        "name": _safe_name(filename),
        "content_type": content_type or "application/octet-stream",
        "size_bytes": len(payload),
        "file": audio_path.name,
    }
    audio_path.write_bytes(payload)
    _metadata_path(reference_id).write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    return meta


def list_all() -> list[dict]:
    items = []
    for meta_path in sorted(storage_dir().glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (storage_dir() / meta["file"]).is_file():
                items.append(meta)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return items


def get(reference_id: str) -> tuple[dict, Path] | None:
    if not re.fullmatch(r"[a-f0-9]{32}", reference_id):
        return None
    try:
        meta = json.loads(_metadata_path(reference_id).read_text(encoding="utf-8"))
        audio_path = storage_dir() / meta["file"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if audio_path.parent != storage_dir() or not audio_path.is_file():
        return None
    return meta, audio_path


def delete(reference_id: str) -> bool:
    found = get(reference_id)
    if found is None:
        return False
    _, audio_path = found
    audio_path.unlink(missing_ok=True)
    _metadata_path(reference_id).unlink(missing_ok=True)
    return True
