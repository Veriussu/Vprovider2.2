# ─────────────────────────────────────────────────────────────
#  Bölüm:    Müzik (Music) API ve Eklenti Köprüsü Testleri
#  Dosya:    tests/test_music_api.py
#  Amaç:     /v1/audio/music/generate ucunu ve çalışan müzik
#            eklentisine (MusicGen/Stable Audio) yönlendirmeyi doğrular.
#  Mekanik:  - Eklenti portu ve HTTP POST monkeypatch ile taklit edilir.
#            - Eklenti yoksa yerel placeholder'a düşülür.
# ─────────────────────────────────────────────────────────────

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import auth, music_backend, plugins
from app import main as main_mod
from app.config import settings
from app.user_store import UserStore

TEST_API_KEY = "muzik-test-anahtar"


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture()
def client(tmp_path, monkeypatch):
    store = UserStore(tmp_path / "music.db")
    store.init()
    store.create_api_key("Müzik Test", TEST_API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    return TestClient(main_mod.app)


def _auth():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


def test_generate_music_via_plugin(monkeypatch):
    """Çalışan müzik eklentisi varsa üretim ona yönlenir (WAV)."""
    import base64

    monkeypatch.setattr(settings, "music_enabled", True)
    monkeypatch.setattr(
        plugins, "running_port_for_category",
        lambda category, prefer=(): ("musicgen_small", 19300),
    )
    wav = b"RIFFxxxxWAVE-plugin-wav"

    async def fake_post(url, payload):
        assert url == "http://127.0.0.1:19300/generate"
        assert payload["prompt"] == "caz"
        return _FakeResp(200, {"audio": base64.b64encode(wav).decode(), "sample_rate": 32000})

    monkeypatch.setattr(music_backend, "_post_json", fake_post)
    data, mime = asyncio.run(music_backend.generate_music("caz", duration=5))
    assert data == wav
    assert mime == "audio/wav"


def test_generate_music_falls_back_without_plugin(monkeypatch):
    """Eklenti yoksa yerel placeholder üretimi kullanılır."""
    monkeypatch.setattr(settings, "music_enabled", True)
    monkeypatch.setattr(plugins, "running_port_for_category", lambda category, prefer=(): None)
    data, mime = asyncio.run(music_backend.generate_music("caz", duration=1.0))
    assert data[:4] == b"RIFF"
    assert mime == "audio/wav"


def test_generate_music_plugin_error(monkeypatch):
    """Eklenti hata dönerse MusicError'a ayrıntı taşınır."""
    monkeypatch.setattr(settings, "music_enabled", True)
    monkeypatch.setattr(
        plugins, "running_port_for_category",
        lambda category, prefer=(): ("musicgen_small", 19300),
    )

    async def fake_post(url, payload):
        return _FakeResp(500, {"error": "bellek bitti"})

    monkeypatch.setattr(music_backend, "_post_json", fake_post)
    with pytest.raises(music_backend.MusicError) as exc:
        asyncio.run(music_backend.generate_music("caz", duration=5))
    assert "bellek bitti" in str(exc.value)


def test_music_endpoint_requires_api_key(client):
    assert client.post("/v1/audio/music/generate", json={"prompt": "caz"}).status_code == 401


def test_music_endpoint_via_plugin_returns_wav(client, monkeypatch):
    """API ucu eklenti üzerinden WAV döner."""
    import base64

    monkeypatch.setattr(settings, "music_enabled", True)
    monkeypatch.setattr(
        plugins, "running_port_for_category",
        lambda category, prefer=(): ("musicgen_small", 19300),
    )
    wav = b"RIFFxxxxWAVE-endpoint-wav"

    async def fake_post(url, payload):
        return _FakeResp(200, {"audio": base64.b64encode(wav).decode(), "sample_rate": 32000})

    monkeypatch.setattr(music_backend, "_post_json", fake_post)
    r = client.post("/v1/audio/music/generate", json={"prompt": "caz"}, headers=_auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content == wav
