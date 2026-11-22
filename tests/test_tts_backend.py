# ─────────────────────────────────────────────────────────────
#  Bölüm:    Ses Üretim Motoru Testleri
#  Dosya:    tests/test_tts_backend.py
#  Amaç:     tts_backend.synthesize + depo akışını sahte edge-tts ile
#            doğrular; gerçek Microsoft servisine istek atılmaz.
#  Mekanik:  - edge_tts modülü sys.modules'a sahte nesneyle enjekte edilir.
#            - Modül kapalı / boş metin / uzun metin / bilinmeyen motor
#              hata senaryoları ayrı ayrı denetlenir.
# ─────────────────────────────────────────────────────────────

import asyncio
import sys

import pytest

from app import tts_backend as tts
from app.tts_backend import TTSError


class FakeAudioChunk:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return {"type": "audio", "data": self._data}[key]


class FakeCommunicate:
    def __init__(self, text, voice, rate=""):
        self.text = text
        self.voice = voice
        self.rate = rate

    async def stream(self):
        yield {"type": "audio", "data": b"ID3-sahte-mp3-1"}
        yield {"type": "word-boundary", "text": "x", "offset": 0, "duration": 1}
        yield {"type": "audio", "data": b"-2"}


class FakeEdgeModule:
    Communicate = FakeCommunicate


@pytest.fixture()
def fake_edge(monkeypatch):
    """edge_tts modülünü sahte sürümle değiştirir."""
    monkeypatch.setitem(sys.modules, "edge_tts", FakeEdgeModule)
    monkeypatch.setattr(tts.settings, "tts_enabled", True)


def test_synthesize_edge_returns_bytes(fake_edge):
    """Sahte edge-tts ile konuşma baytları üretilir."""
    data, mime = asyncio.run(tts.synthesize("merhaba dünya", voice="tr-TR-EmelNeural"))
    assert data == b"ID3-sahte-mp3-1-2"
    assert mime == "audio/mpeg"


def test_synthesize_disabled_raises():
    """Modül kapalıyken anlaşılır hata verilir."""
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("merhaba"))
    assert "kapalı" in str(exc.value)


def test_synthesize_empty_text_raises(fake_edge):
    """Boş metin reddedilir."""
    with pytest.raises(TTSError):
        asyncio.run(tts.synthesize("   "))


def test_synthesize_too_long_raises(fake_edge):
    """Karakter üst sınırı aşılınca reddedilir."""
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("a" * (tts.MAX_TEXT_CHARS + 1)))
    assert "çok uzun" in str(exc.value)


def test_synthesize_unknown_engine_raises(fake_edge):
    """Bilinmeyen motor reddedilir."""
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("merhaba", engine="hayalet-motor"))
    assert "hayalet-motor" in str(exc.value)


def test_engine_available_with_fake(fake_edge):
    """Sahte modül varken motor kurulu kabul edilir."""
    assert tts.engine_available() is True


def test_edge_speed_rate():
    """OpenAI speed değeri edge-tts rate biçimine çevrilir."""
    assert tts._edge_speed_rate(1.0) == "+0%"
    assert tts._edge_speed_rate(1.1) == "+10%"
    assert tts._edge_speed_rate(0.85) == "-15%"


def test_audio_store_roundtrip():
    """Depoya yazılan ses okunabilir ve cap uygulanır."""
    tts.store_audio("k1", b"AAA", "audio/mpeg")
    assert tts.get_stored_audio("k1") == {"bytes": b"AAA", "mime": "audio/mpeg"}
    assert tts.get_stored_audio("yok") is None
    # cap: en son üretimler korunur
    for i in range(tts.AUDIO_STORE_CAP + 5):
        tts.store_audio(f"k{i}", b"X", "audio/mpeg")
    assert tts.get_stored_audio("k1") is None  # eski silindi
    assert tts.get_stored_audio(f"k{tts.AUDIO_STORE_CAP + 4}") is not None


# ------------------------------------------------------------------
# Eklenti (plugin) motoru köprüsü
# ------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _wav(seconds=1.0, sr=24000) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    return buf.getvalue()


def test_language_from_voice():
    """'voice' alanından XTTS dil kodu çıkarılır."""
    assert tts._language_from_voice("tr-TR-EmelNeural") == "tr"
    assert tts._language_from_voice("en") == "en"
    assert tts._language_from_voice("") == "en"
    assert tts._language_from_voice("Emel") == "en"


def test_wav_duration_sec():
    """WAV süresi başlıktan doğru okunur; geçersiz veri 0 döner."""
    assert tts.wav_duration_sec(_wav(2.0, 16000)) == pytest.approx(2.0, abs=0.05)
    assert tts.wav_duration_sec(b"not-a-wav") == 0.0


def test_synthesize_plugin_ok(monkeypatch):
    """Eklenti çalışırken synthesize plugin motoruyla WAV döner."""
    import base64

    from app import plugins

    monkeypatch.setattr(tts.settings, "tts_enabled", True)
    monkeypatch.setattr(
        plugins, "running_port_for_category",
        lambda category, prefer=(): ("xtts_v2", 19999),
    )
    wav = _wav(1.0)

    async def fake_post(url, payload):
        assert url == "http://127.0.0.1:19999/synthesize"
        assert payload["text"] == "merhaba"
        assert payload["language"] == "tr"
        return _FakeResp(200, {"audio": base64.b64encode(wav).decode(), "sample_rate": 24000})

    monkeypatch.setattr(tts, "_post_json", fake_post)
    data, mime = asyncio.run(
        tts.synthesize("merhaba", engine="plugin", voice="tr-TR-EmelNeural")
    )
    assert data == wav
    assert mime == "audio/wav"


def test_synthesize_plugin_not_running(monkeypatch):
    """Çalışan TTS eklentisi yoksa anlaşılır hata verilir."""
    from app import plugins

    monkeypatch.setattr(tts.settings, "tts_enabled", True)
    monkeypatch.setattr(plugins, "running_port_for_category", lambda category, prefer=(): None)
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("merhaba", engine="plugin"))
    assert "çalışmıyor" in str(exc.value)


def test_synthesize_plugin_http_error(monkeypatch):
    """Eklenti 500 dönerse hata ayrıntısı taşınır."""
    from app import plugins

    monkeypatch.setattr(tts.settings, "tts_enabled", True)
    monkeypatch.setattr(
        plugins, "running_port_for_category",
        lambda category, prefer=(): ("xtts_v2", 19999),
    )

    async def fake_post(url, payload):
        return _FakeResp(500, {"error": "model patladı"})

    monkeypatch.setattr(tts, "_post_json", fake_post)
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("merhaba", engine="plugin"))
    assert "model patladı" in str(exc.value)