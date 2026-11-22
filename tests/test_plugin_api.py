# ─────────────────────────────────────────────────────────────
#  Bölüm:    Eklenti Köprü Uçları Testleri
#  Dosya:    tests/test_plugin_api.py
#  Amaç:     /v1/images/upscale, /v1/videos/restore, /v1/videos/lipsync
#            uçlarının eklenti servisine yönlendirmesini sahte HTTP ile
#            doğrular (gerçek eklenti kurulu değildir).
#  Mekanik:  - plugins.running_port_for_category sahtelenir.
#            - plugin_api._http_post_json sahtelenir (ağ yok).
#            - Çalışan eklenti yoksa 503 (plugin_not_running).
#            - Eklenti hatası 502; 401/400 durumları denetlenir.
# ─────────────────────────────────────────────────────────────

import httpx
import pytest
from fastapi.testclient import TestClient

from app import auth, plugin_api, plugins
from app import main as main_mod
from app.gateway import hash_key
from app.user_store import UserStore

TEST_API_KEY = "eklenti-test-anahtar"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API anahtarlı hazır test ortamı; eklenti yok (varsayılan)."""
    store = UserStore(tmp_path / "plugin.db")
    store.init()
    store.create_api_key("Eklenti Test", TEST_API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    monkeypatch.setattr(plugins, "running_port_for_category", lambda *a, **k: None)
    return TestClient(main_mod.app)


def _auth():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_post(monkeypatch, response):
    """_http_post_json'i sahteler ve çağrıları kaydeder."""
    calls = {}

    async def fake(url, payload):
        calls["url"] = url
        calls["payload"] = payload
        return response

    monkeypatch.setattr(plugin_api, "_http_post_json", fake)
    return calls


def _running(monkeypatch, plugin_id="real_esrgan", port=1234):
    monkeypatch.setattr(
        plugins, "running_port_for_category", lambda *a, **k: (plugin_id, port)
    )


# ------------------------------------------------------------------
# Kimlik doğrulama
# ------------------------------------------------------------------

def test_upscale_requires_api_key(client):
    assert client.post("/v1/images/upscale", json={"image": "x"}).status_code == 401
    assert client.post("/v1/videos/restore", json={"video": "x"}).status_code == 401
    assert client.post("/v1/videos/lipsync", json={"audio": "x", "video": "x"}).status_code == 401


# ------------------------------------------------------------------
# Çalışan eklenti yok -> 503
# ------------------------------------------------------------------

def test_upscale_without_running_plugin_returns_503(client):
    r = client.post("/v1/images/upscale", json={"image": "x"}, headers=_auth())
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "plugin_not_running"


# ------------------------------------------------------------------
# Başarılı yönlendirme
# ------------------------------------------------------------------

def test_upscale_proxies_to_plugin(client, monkeypatch):
    _running(monkeypatch, "real_esrgan", 5001)
    calls = _fake_post(monkeypatch, _FakeResponse(200, {"image": "OUT"}))

    r = client.post("/v1/images/upscale", json={"image": "IN"}, headers=_auth())

    assert r.status_code == 200
    assert r.json() == {"image": "OUT"}
    assert calls["url"] == "http://127.0.0.1:5001/upscale"
    assert calls["payload"] == {"image": "IN"}


def test_restore_proxies_to_plugin(client, monkeypatch):
    _running(monkeypatch, "seedvr2_3b", 5002)
    calls = _fake_post(monkeypatch, _FakeResponse(200, {"video": "OUTV"}))

    r = client.post("/v1/videos/restore", json={"video": "INV"}, headers=_auth())

    assert r.status_code == 200
    assert r.json() == {"video": "OUTV"}
    assert calls["url"] == "http://127.0.0.1:5002/restore"
    assert calls["payload"] == {"video": "INV"}


def test_lipsync_proxies_to_plugin(client, monkeypatch):
    _running(monkeypatch, "latent_sync", 5003)
    calls = _fake_post(monkeypatch, _FakeResponse(200, {"video": "SYNCED"}))

    r = client.post(
        "/v1/videos/lipsync",
        json={"audio": "AUD", "image": "IMG"},
        headers=_auth(),
    )

    assert r.status_code == 200
    assert calls["url"] == "http://127.0.0.1:5003/lipsync"
    assert calls["payload"] == {"audio": "AUD", "image": "IMG"}


def test_lipsync_requires_video_or_image(client):
    # Girdi denetimi çalışan eklenti aranmadan önce yapılır.
    r = client.post("/v1/videos/lipsync", json={"audio": "AUD"}, headers=_auth())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_input"


# ------------------------------------------------------------------
# Hata yolları
# ------------------------------------------------------------------

def test_plugin_error_maps_to_502(client, monkeypatch):
    _running(monkeypatch, "real_esrgan", 5001)
    _fake_post(monkeypatch, _FakeResponse(500, {"error": "patladı"}))

    r = client.post("/v1/images/upscale", json={"image": "IN"}, headers=_auth())

    assert r.status_code == 502
    assert r.json()["error"]["code"] == "plugin_error"


def test_plugin_unreachable_maps_to_502(client, monkeypatch):
    _running(monkeypatch, "real_esrgan", 5001)

    async def boom(url, payload):
        raise httpx.ConnectError("bağlantı yok")

    monkeypatch.setattr(plugin_api, "_http_post_json", boom)

    r = client.post("/v1/images/upscale", json={"image": "IN"}, headers=_auth())

    assert r.status_code == 502
    assert r.json()["error"]["code"] == "plugin_unreachable"


def test_plugin_bad_json_maps_to_502(client, monkeypatch):
    _running(monkeypatch, "real_esrgan", 5001)
    _fake_post(monkeypatch, _FakeResponse(200, ValueError("geçersiz json")))

    r = client.post("/v1/images/upscale", json={"image": "IN"}, headers=_auth())

    assert r.status_code == 502
    assert r.json()["error"]["code"] == "plugin_bad_response"


# ------------------------------------------------------------------
# Tarife (pricing) — köprü işleri kredi fiyatlandırması
# ------------------------------------------------------------------

def test_bridge_tasks_are_priced():
    from app.pricing import compute_cost, get_prices

    assert get_prices("upscale")["per_image"] > 0
    assert get_prices("restore")["per_video"] > 0
    assert get_prices("lipsync")["per_video"] > 0
    assert compute_cost("upscale", images=1) == pytest.approx(0.02)
    assert compute_cost("lipsync", videos=1) == pytest.approx(0.3)


def test_upscale_model_allowlist_enforced(tmp_path, monkeypatch):
    """Anahtarın izin listesi köprü işlerine de uygulanır (403)."""
    store = UserStore(tmp_path / "plugin2.db")
    store.init()
    store.create_api_key("Eklenti Test", TEST_API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    monkeypatch.setattr(plugins, "running_port_for_category", lambda *a, **k: ("real_esrgan", 5001))
    _fake_post(monkeypatch, _FakeResponse(200, {"image": "OUT"}))
    key_id = store.get_api_key_by_hash(hash_key(TEST_API_KEY))["id"]
    store.update_api_key(key_id, model_allowlist=["başka"])

    client = TestClient(main_mod.app)
    r = client.post("/v1/images/upscale", json={"image": "IN"}, headers=_auth())

    assert r.status_code == 403
    assert r.json()["error"]["code"] == "model_not_allowed"
