# ─────────────────────────────────────────────────────────────
#  Bölüm:    Ses Üretimi (TTS) Motoru
#  Dosya:    app/tts_backend.py
#  Amaç:     Metinden konuşma (TTS) üretir. Varsayılan motor edge-tts
#            (Microsoft Edge çevrimiçi motoru): hafif, saf Python, MP3
#            üretir; GPU gerekmez. Panel üretimleri in-memory depoda
#            tutulur (çoktan-azan düzen).
#  Not:      Modül kapalıysa ya da metin çok uzunsa TTSError verilir;
#            API katmanı bunu OpenAI uyumlu hataya çevirir.
#  Kullanım: rez = await synthesize("merhaba", voice="tr-TR-EmelNeural")
#            rez -> (ses_baytları, mime)
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import logging
import threading
import time
from typing import Optional

from app.config import settings
from app.tts_text import normalize_tts_text

logger = logging.getLogger("vprovider")

MAX_TEXT_CHARS = 3000          # edge-tts üst sınırı (~3000 karakter)
TTS_TIMEOUT = 120              # tek konuşma için üst süre (saniye)
AUDIO_STORE_CAP = 20           # depoda tutulacak son üretim sayısı

# Panel seçiminde sunulan Türkçe sesler (edge-tts'te en yaygın kullanılanlar)
TR_VOICES: dict[str, str] = {
    "tr-TR-EmelNeural": "Emel (kadın)",
    "tr-TR-AhmetNeural": "Ahmet (erkek)",
}


class TTSError(RuntimeError):
    """Ses üretim hataları (kapalı modül, kurulu olmayan motor, uzun metin...)."""


# ------------------------------------------------------------------
# Ses deposu (in-memory): panel arayüzünce sunulan son üretimler
# ------------------------------------------------------------------

_audio_store: dict[str, dict] = {}
_audio_store_lock = threading.Lock()


def store_audio(key: str, audio: bytes, mime: str) -> None:
    """Üretilen sesi çoktan-azan düzen (cap) ile saklar."""
    with _audio_store_lock:
        _audio_store[key] = {"bytes": audio, "mime": mime}
        while len(_audio_store) > AUDIO_STORE_CAP:
            _audio_store.pop(next(iter(_audio_store)))


def get_stored_audio(key: str) -> Optional[dict]:
    """Depodan ses baytlarını döner (yoksa None)."""
    with _audio_store_lock:
        return _audio_store.get(key)


# ------------------------------------------------------------------
# Motor
# ------------------------------------------------------------------

def _edge_communicate(text: str, voice: str, rate: str):
    """edge-tts.Communicate örneği üretir.

    Ayrı fonksiyona çıkarıldı ki testler sahte modülle yamalayabilsin.
    """
    import edge_tts  # işlev anında yüklenir; bağımlılık isteğe bağlıdır

    return edge_tts.Communicate(text, voice, rate=rate)


def _import_edge():
    import edge_tts  # noqa: F401

    return edge_tts


async def _synthesize_edge(text: str, voice: str, rate: str) -> tuple[bytes, str]:
    """edge-tts ile konuşmayı üretir -> (mp3_baytları, "audio/mpeg")."""
    try:
        communicate = await asyncio.to_thread(_edge_communicate, text, voice, rate)
    except ImportError:
        raise TTSError(
            "edge-tts motoru kurulu değil. Kurun: pip install edge-tts "
            "(veya venv'inize requirements.txt'i yükleyin)."
        )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    if buf.tell() == 0:
        raise TTSError("edge-tts ses üretemedi (ağ/geçersiz ses adı).")
    return buf.getvalue(), "audio/mpeg"


async def _post_json(url: str, payload: dict):
    """Eklenti servisine JSON POST atar (test için ayrı tutulur)."""
    import httpx

    async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
        return await client.post(url, json=payload)


async def _synthesize_plugin(
    text: str,
    language: str = "en",
    reference_path: str = "",
    reference_b64: str = "",
) -> tuple[bytes, str]:
    """Çalışan TTS eklentisiyle (XTTS-v2) konuşma üretir -> (wav, "audio/wav")."""
    import base64

    from app import plugins
    from app.runtime_coord import accelerator

    found = plugins.running_port_for_category("tts", prefer=("xtts_v2",))
    if found is None:
        raise TTSError(
            "TTS eklentisi çalışmıyor; XTTS-v2'yi Panel > Eklentiler'den başlatın."
        )
    _plugin_id, port = found

    payload: dict = {"text": text, "language": language or "en"}
    if reference_path:
        payload["reference_path"] = reference_path
    if reference_b64:
        payload["reference_b64"] = reference_b64

    url = f"http://127.0.0.1:{port}/synthesize"
    async with accelerator.acquire("plugin:tts"):
        try:
            resp = await _post_json(url, payload)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"TTS eklentisine ulaşılamadı: {exc}") from exc

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise TTSError(f"TTS eklenti hatası: {detail}")

    data = resp.json()
    audio_b64 = data.get("audio")
    if not audio_b64:
        raise TTSError("TTS eklentisi ses döndürmedi.")
    return base64.b64decode(audio_b64), "audio/wav"


async def synthesize(
    text: str,
    voice: str = "",
    engine: str = "",
    rate: str = "",
    language: str = "",
) -> tuple[bytes, str]:
    """Metni seslendirir.

    Dönüş: (ses_baytları, mime). Yanlış/kapalı kurulumda TTSError yükseltir.
    """
    if not settings.tts_enabled:
        raise TTSError("Ses üretim modülü kapalı (TTS_ENABLED=false).")

    text = normalize_tts_text(text)
    if not text:
        raise TTSError("Konuşulacak metin boş olamaz.")
    if len(text) > MAX_TEXT_CHARS:
        raise TTSError(f"Metin çok uzun: en fazla {MAX_TEXT_CHARS} karakter olabilir.")

    engine = engine or settings.tts_engine
    if engine == "edge":
        voice = voice or settings.tts_voice
        rate = rate or settings.tts_voice_rate
        return await _synthesize_edge(text, voice, rate)

    if engine in ("plugin", "xtts"):
        return await _synthesize_plugin(
            text,
            language=language or _language_from_voice(voice),
        )

    raise TTSError(f"Bilinmeyen ses motoru: {engine} (kullanılabilir: edge, plugin).")


def _language_from_voice(voice: str) -> str:
    """OpenAI 'voice' alanından XTTS dil kodu çıkarır (ör. 'tr-TR-...' -> 'tr')."""
    if not voice:
        return "en"
    code = voice.split("-")[0].strip().lower()
    if len(code) == 2 and code.isalpha():
        return code
    return "en"


def wav_duration_sec(wav: bytes) -> float:
    """16-bit PCM WAV baytlarından süreyi okur (byte -> saniye)."""
    import struct

    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return 0.0
    try:
        sample_rate = struct.unpack("<I", wav[24:28])[0]
        byte_rate = struct.unpack("<I", wav[28:32])[0]
    except struct.error:
        return 0.0
    if not sample_rate or not byte_rate:
        return 0.0
    return round(max(0, len(wav) - 44) / byte_rate, 2)


def engine_available() -> bool:
    """Seçili motorun bellekte yüklü olup olmadığını döner (kurulum denetimi)."""
    try:
        _import_edge()
        return True
    except ImportError:
        return False


def _edge_speed_rate(speed: float) -> str:
    """OpenAI speed değerini (1.0 = normal) edge-tts 'rate' biçimine çevirir."""
    pct = int(round((speed - 1.0) * 100))
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct}%"


async def tts_status() -> dict:
    """Panel için durum bilgisi: açık mı, motor kurulu mu, hangi ses kullanılıyor."""
    return {
        "enabled": settings.tts_enabled,
        "engine": settings.tts_engine,
        "voice": settings.tts_voice,
        "engine_installed": engine_available() if settings.tts_enabled else None,
        "rate": settings.tts_voice_rate,
    }