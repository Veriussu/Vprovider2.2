# ─────────────────────────────────────────────────────────────
#  Bölüm:    STT (Speech-to-Text) Backend
#  Amaç:     faster-whisper (ctranslate2) ile ses metne dönüştürme
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path
import threading
import time
from typing import Optional, Tuple

import numpy as np
from app.config import settings
from app.runtime_coord import accelerator

logger = logging.getLogger("vprovider")

MAX_AUDIO_SECONDS = 300  # Tek seferde işlenecek maksimum audio süresi (saniye)
STT_STORE_CAP = 20       # Panel deposunda tutulacak son transkripsiyon sayısı

# Panel seçiminde sunulan ses dosyaları için klasör
_AUDIO_STORE: dict[str, dict] = {}
_AUDIO_STORE_LOCK = threading.Lock()


class STTError(RuntimeError):
    """Ses-to-Text hataları (model yüklenemedi, ses bozuk, vb.)."""


def _store_transcription(key: str, text: str, language: str = "tr") -> None:
    """Transkripsiyonu çoktan-azan düzenle (LRU) saklar."""
    with _AUDIO_STORE_LOCK:
        _AUDIO_STORE[key] = {"text": text, "language": language, "timestamp": time.time()}
        while len(_AUDIO_STORE) > STT_STORE_CAP:
            _AUDIO_STORE.pop(next(iter(_AUDIO_STORE)))


def _get_stored_transcription(key: str) -> Optional[dict]:
    """Depodan transkripsiyonu döner (yoksa None)."""
    with _AUDIO_STORE_LOCK:
        return _AUDIO_STORE.get(key)


def _ensure_model_dir(model_name: str) -> str:
    """Modelin yerel yolu döner; yoksa HuggingFace'ten indirir."""
    model_path = settings.models_path / "ct2" / model_name
    if model_path.exists() and (model_path / "model.bin").exists():
        return str(model_path)

    default_model = settings.models_path / "ct2" / "default"
    if default_model.exists() and (default_model / "model.bin").exists():
        return str(default_model)

    logger.info("STT modeli indiriliyor: %s", model_name)
    try:
        from faster_whisper.utils import download_model

        downloaded = download_model(model_name, output_dir=str(model_path))
        if (Path(downloaded) / "model.bin").exists():
            return str(downloaded)
        if (model_path / "model.bin").exists():
            return str(model_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("STT modeli indirilemedi: %s", model_name)
        raise STTError(f"STT modeli indirilemedi: {exc}") from exc

    raise STTError(f"STT modeli bulunamadı: {model_name}")


async def transcribe_audio(
    audio_bytes: bytes,
    model: str = "base",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: list = ["segment"],
) -> dict:
    """Ses verisini metne çevirir (OpenAI uyumlu).

    GPU işi (model yükleme + çıkarım) VProvider geneli tek iş kuyruğundan
    geçer; aynı anda başka bir model işi çalışmaz.
    """
    async with accelerator.acquire(f"stt:{model}"):
        return await asyncio.to_thread(
            _transcribe_core,
            audio_bytes, model, language, prompt, temperature,
            response_format, timestamp_granularities,
        )


def _transcribe_core(
    audio_bytes: bytes,
    model: str = "base",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: list = ["segment"],
) -> dict:
    """
    Ses verisini metne çevirir (faster-whisper kullanarak).
    
    Args:
        audio_bytes: WAV formatında ses verisi
        model: ct2 model adı (base, small, medium, large-v2, vb.)
        language: Dil kodu (örn: "tr", "en") veya None (otomatik tespit)
        prompt: İlk parti için ipucu metni
        temperature: Örnekleme sıcaklığı (0.0 = deterministik)
        response_format: "json", "text", "srt", "vtt", "tsv"
        timestamp_granularities: ["word", "segment"] listesi
    
    Returns:
        OpenAI uyumlu transcription yanıtı
    """
    if not settings.stt_enabled:
        raise STTError("STT modülü kapalı (STT_ENABLED=false).")

    if not audio_bytes:
        raise STTError("Ses verisi boş olamaz.")

    # faster-whisper modelini yükle (gerekirse indir)
    try:
        from faster_whisper import WhisperModel

        model_path = _ensure_model_dir(model)

        if not hasattr(_transcribe_core, "_model") or _transcribe_core._model is None or _transcribe_core._model_path != model_path:
            logger.info("STT modeli yükleniyor: %s", model_path)
            device = "cuda" if settings.gpu_mode != "cpu" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            _transcribe_core._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
            )
            _transcribe_core._model_path = model_path

    except ImportError as exc:
        raise STTError(
            "faster-whisper kurulu değil. Kurun: pip install faster-whisper "
            "(veya requirements-hub.txt'yi yükleyin)."
        ) from exc
    except Exception as exc:
        logger.error("STT modeli yüklenemedi: %s", exc)
        raise STTError(f"STT modeli yüklenemedi: {exc}") from exc

    # Ses dosyasını geçici olarak kaydet; librosa/ffmpeg ile 16 kHz mono'ya çevir.
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    try:
        try:
            import librosa
        except ImportError as exc:
            raise STTError("librosa kurulu değil; ses çözülemedi.") from exc

        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        if sample_rate != 16000:
            raise STTError("Ses 16 kHz'e dönüştürülemedi.")
        duration_seconds = len(audio) / sample_rate
        if duration_seconds > MAX_AUDIO_SECONDS:
            raise STTError(f"Ses çok uzun: maksimum {MAX_AUDIO_SECONDS} saniye.")

        # Transkripsiyonu çalıştır
        segments, info = _transcribe_core._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # Sonuçları topla
        segments_list = []
        full_text = []
        
        for segment in segments:
            segment_dict = {
                "id": len(segments_list),
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }
            segments_list.append(segment_dict)
            full_text.append(segment.text)
        
        text = "".join(full_text).strip()

        if getattr(info, "duration", 0) and info.duration > MAX_AUDIO_SECONDS:
            raise STTError(f"Ses çok uzun: maksimum {MAX_AUDIO_SECONDS} saniye.")

        # Yanıtı OpenAI formatında hazırla
        result = {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": segments_list,
        }
        
        # response_format'a göre çıktıyı ayarla
        if response_format == "text":
            return text
        elif response_format in ["srt", "vtt", "tsv"]:
            # Basit implementasyon - gerçekte daha kapsamlı olmalı
            if response_format == "srt":
                srt_lines = []
                for i, seg in enumerate(segments_list, 1):
                    start = seg["start"]
                    end = seg["end"]
                    srt_lines.append(
                        f"{i}\n"
                        f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d} --> "
                        f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}\n"
                        f"{seg['text']}\n"
                    )
                return "\n".join(srt_lines)
            elif response_format == "vtt":
                vtt_lines = ["WEBVTT\n"]
                for seg in segments_list:
                    start = seg["start"]
                    end = seg["end"]
                    vtt_lines.append(
                        f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d}.{int((start%1)*1000):03d} --> "
                        f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d}.{int((end%1)*1000):03d}\n"
                        f"{seg['text']}\n"
                    )
                return "\n".join(vtt_lines)
            elif response_format == "tsv":
                tsv_lines = ["start\tend\ttext"]
                for seg in segments_list:
                    tsv_lines.append(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['text']}")
                return "\n".join(tsv_lines)
        
        # JSON formatı için tüm segmentleri ekle
        if response_format == "json":
            result["segments"] = segments_list
            return result
        
        # Varsayılan JSON
        return result
        
    except Exception as e:
        logger.error(f"STT transkripsiyonu başarısız: {e}")
        raise STTError(f"Transkripsiyon başarısız: {e}")
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


async def store_transcription(key: str, text: str, language: str = "tr") -> None:
    """Transkripsiyonu depoda saklar (panel için)."""
    _store_transcription(key, text, language)


async def get_stored_transcription(key: str) -> Optional[dict]:
    """Depodan transkripsiyonu döner."""
    return _get_stored_transcription(key)


async def stt_status() -> dict:
    """Panel için STT durum bilgisi."""
    try:
        from faster_whisper import WhisperModel
        faster_whisper_available = True
    except ImportError:
        faster_whisper_available = False
    
    return {
        "enabled": settings.stt_enabled,
        "faster_whisper_installed": faster_whisper_available,
        "default_model": getattr(settings, "stt_model", "base"),
        "model_dir_exists": (settings.models_path / "ct2").exists(),
    }


# Backwards compatibility alias
speech_to_text = transcribe_audio
