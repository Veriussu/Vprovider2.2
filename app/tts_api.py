# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu Ses (TTS) API'si
#  Dosya:    app/tts_api.py
#  Amaç:     OpenAI /v1/audio/speech ile uyumlu metin-sesleme uçları sunar.
#            OpenAI bu isteklerde sesi ham bayt (audio/mpeg) döner; aynı
#            davranış korunur. kimlik doğrulama (Bearer) ve hata yapısı
#            (detail.error) tüm /v1/* uçlarıyla aynıdır.
#  Mekanik:  - POST /v1/audio/speech  -> {model, input, voice, response_format}
#              ham MP3 döner.
#            - GET  /v1/audio/voices   -> desteklenen Türkçe sesler (panel için)
#  Kullanım: app/main.py içinde app.include_router(tts_router)
# ─────────────────────────────────────────────────────────────

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.routing import APIRouter as _Router
from pydantic import BaseModel, Field

from app import tts_backend
from app.config import settings
from app.gateway import ApiContext, api_auth
from app.openai_api import _openai_error
from app.tts_backend import TTSError, tts_status

router: _Router = APIRouter(prefix="/v1/audio", dependencies=[Depends(api_auth)])


# ------------------------------------------------------------------
# İstek modeli (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """POST /v1/audio/speech istek gövdesi (OpenAI ile aynı alanlar)."""

    model: str = "edge-tts"       # motor adı (edge|edge-tts → edge, xtts|plugin → eklenti)
    input: str = Field(..., min_length=1)
    voice: str = ""               # boşsa TTS_VOICE kullanılır
    language: str = Field("", description="XTTS dili (ör. tr, en); boşsa 'voice'tan çıkarılır")
    response_format: str = "mp3"  # mp3 (edge-tts) / wav (plugin)
    speed: float = Field(1.0, ge=0.5, le=2.0)  # 1.0 = normal hız


def _resolve_engine(model: str) -> tuple[str, bool]:
    """Model adından (motor, eklenti mi?) döner."""
    lowered = (model or "").lower()
    if lowered in ("xtts", "xtts-v2", "xtts_v2", "plugin", "coqui/xtts-v2"):
        return "plugin", True
    return "edge", False


# ------------------------------------------------------------------
# Uç noktalar
# ------------------------------------------------------------------

@router.post("/speech")
async def create_speech(req: SpeechRequest, ctx: ApiContext = Depends(api_auth)):
    """Metni seslendirir; ham ses (audio/mpeg | audio/wav) döner (OpenAI ile aynı)."""
    engine, is_plugin = _resolve_engine(req.model)
    if is_plugin and req.response_format not in ("wav", "mp3"):
        raise _openai_error(
            400,
            "TTS eklentisi wav üretir; response_format='wav' kullanın.",
            param="response_format",
        )
    if not is_plugin and req.response_format != "mp3":
        raise _openai_error(
            400,
            "edge-tts yalnızca mp3 üretir; response_format='mp3' kullanın.",
            param="response_format",
        )
    try:
        audio, mime = await tts_backend.synthesize(
            text=req.input,
            voice=req.voice,
            engine=engine,
            rate=tts_backend._edge_speed_rate(req.speed),
            language=req.language,
        )
    except TTSError as exc:
        raise _openai_error(400, str(exc), code="speech_failed")
    except Exception as exc:
        raise _openai_error(502, f"Ses motoru bağlantı hatası: {exc}", code="tts_unreachable")

    # Süre kestirimi: eklenti WAV döner (başlıktan okur); edge 128 kbps MP3.
    if is_plugin:
        audio_sec = tts_backend.wav_duration_sec(audio)
    else:
        audio_sec = round(len(audio) * 8 / 128_000, 2)
    ctx.charge("tts", model_id=req.model, audio_sec=audio_sec)
    return Response(content=audio, media_type=mime)


@router.get("/voices")
async def list_voices():
    """Desteklenen Türkçe sesleri listeler (OpenAI'de yok; VProvider eklentisi)."""
    return {
        "object": "list",
        "data": [
            {"id": vid, "label": label, "locale": "tr-TR"}
            for vid, label in tts_backend.TR_VOICES.items()
        ],
    }


@router.get("/status")
async def status_endpoint():
    """Motor durumu (açık mı, kurulu mu, hangi ses) — panel ve teşhis için."""
    return await tts_status()