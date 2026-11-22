# ─────────────────────────────────────────────────────────────
#  Bölüm:    Eklenti (Plugin) Kataloğu, Yaşam Döngüsü ve Panel Uçları
#  Dosya:    tests/test_plugins.py
#  Amaç:     Eklenti kataloğunun bütünlüğünü ve kurulum/süreç
#            (başlat-durdur-yeniden başlat-log-kaldır) davranışını,
#            gerçek git/uv/ağ işlemi yapmadan doğrular.
#  Mekanik:  - user_store/auth singleton'ları geçici örneklere yönlendirilir.
#            - Kurulum ve süreç yönetimi monkeypatch ile taklit edilir;
#              böylece sonuç makineden ve ağdan bağımsız olur.
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import admin_api, auth, plugins
from app import main as main_mod
from app.auth import hash_password
from app.user_store import UserStore

# _client kalıntı denetimini kapatır; gerçek uygulama testleri için sakla.
_ARTIFACTS_PRESENT_REAL = plugins.artifacts_present
_ARTIFACTS_SIZE_REAL = plugins.artifacts_size_mb


class _FakeProc:
    """Gerçek süreç başlatmadan Popen arayüzünü taklit eder."""

    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def kill(self):
        self._alive = False


class _InlineThread:
    """Ayrı iş parçacığı yerine hedefi hemen çalıştırır (yarışı önler)."""

    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def _client(tmp_path, monkeypatch) -> TestClient:
    """Geçici DB + oturumlu panel istemcisi kurar (hiçbir eklenti kurulu değil)."""
    store = UserStore(tmp_path / "plugins.db")
    store.init()
    store.mark_setup_done()
    store.create_user("yonetici", hash_password("sifre-1234"))

    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(admin_api, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    # Makineden bağımsız: hiçbir eklenti kurulu/çalışır değil.
    monkeypatch.setattr(plugins, "is_installed", lambda recipe: False)
    monkeypatch.setattr(plugins, "is_running", lambda recipe: False)
    # Kalıntı denetimi gerçek disk yollarına (ağırlık hedefleri) bakar; testte
    # makineden bağımsız olması için kapatılır. Kalıntı testleri kendi
    # monkeypatch'iyle bunu geçersiz kılar.
    monkeypatch.setattr(plugins, "artifacts_present", lambda recipe: False)
    monkeypatch.setattr(plugins, "PLUGINS_RUNTIME_DIR", tmp_path / "runtime" / "plugins")
    monkeypatch.setattr(plugins, "UV_DIR", tmp_path / "runtime" / "uv")
    monkeypatch.setattr(plugins, "UV_BIN", tmp_path / "runtime" / "uv" / "uv")
    monkeypatch.setattr(
        plugins, "weights_dir", lambda recipe: tmp_path / "weights" / recipe.id
    )
    monkeypatch.setattr(
        plugins, "venv_dir", lambda recipe: tmp_path / "venvs" / recipe.id
    )
    monkeypatch.setattr(
        plugins, "src_dir", lambda recipe: tmp_path / "src" / recipe.id
    )

    # Global durum kalıntılarını temizle.
    plugins._PROCS.clear()
    plugins._INSTALLS.clear()
    plugins._INSTALL_CANCEL.clear()
    plugins._INSTALL_PROCS.clear()

    return TestClient(main_mod.app)


def _login(client: TestClient) -> None:
    resp = client.post("/panel/login", json={"username": "yonetici", "password": "sifre-1234"})
    assert resp.status_code == 200


def _install_noop(monkeypatch) -> None:
    """Kurulumun arka plan gövdesini gerçek iş yapmadan taklit eder."""
    def fake(recipe):
        with plugins._INSTALL_LOCK:
            current = plugins._INSTALLS.setdefault(recipe.id, {})
            current.update(running=False, done=True, step="tamamlandı", error=None)
    monkeypatch.setattr(plugins, "_do_install", fake)
    monkeypatch.setattr(plugins.threading, "Thread", _InlineThread)


# ------------------------------------------------------------------
# Katalog
# ------------------------------------------------------------------

def test_catalog_contains_expected_plugins():
    ids = {p["id"] for p in plugins.list_plugins()}
    assert ids == {
        "xtts_v2", "muse_talk", "latent_sync", "real_esrgan",
        "seedvr2_3b", "musicgen_small", "stable_audio_open",
    }


def test_catalog_order_matches_request():
    ids = [p["id"] for p in plugins.list_plugins()]
    assert ids == [
        "xtts_v2", "muse_talk", "latent_sync", "real_esrgan",
        "seedvr2_3b", "musicgen_small", "stable_audio_open",
    ]


def test_catalog_entries_have_required_fields():
    for p in plugins.list_plugins():
        for key in ("id", "name", "category", "repo", "description", "python_version", "api"):
            assert p.get(key), f"{p['id']} için '{key}' boş"
        assert p["category"] in {"tts", "lipsync", "upscale", "restore", "music"}
        assert isinstance(p["installed"], bool)
        assert isinstance(p["running"], bool)


def test_every_plugin_has_adapter_or_setup():
    """Her eklenti ya bir servis adaptörü ya da bir kurulum betiği taşımalı."""
    for recipe in plugins.CATALOG:
        assert recipe.adapter or recipe.setup_script, recipe.id


def test_adapters_exist_on_disk():
    """Tariflerdeki adaptör betikleri gerçekten mevcut olmalı."""
    for recipe in plugins.CATALOG:
        if recipe.adapter:
            assert (plugins.ADAPTERS_DIR / recipe.adapter).is_file(), recipe.adapter


def test_get_plugin_unknown_returns_none():
    assert plugins.get_plugin("yok-boyle-eklenti") is None


# ------------------------------------------------------------------
# Eklenti modelleri (Modeller sayfası için)
# ------------------------------------------------------------------

def test_plugin_models_lists_only_installed(tmp_path, monkeypatch):
    """plugin_models yalnızca kurulu eklentileri model kartına çevirir."""
    monkeypatch.setattr(
        plugins, "is_installed", lambda recipe: recipe.id == "xtts_v2"
    )
    monkeypatch.setattr(plugins, "is_running", lambda recipe: recipe.id == "xtts_v2")
    wd = tmp_path / "weights" / "xtts_v2"
    wd.mkdir(parents=True)
    (wd / "model.pth").write_bytes(b"x" * 32)
    monkeypatch.setattr(plugins, "weights_dir", lambda recipe: tmp_path / "weights" / recipe.id)

    models = plugins.plugin_models()

    assert [m["id"] for m in models] == ["xtts_v2"]
    m = models[0]
    assert m["category"] == "tts"
    assert m["kind"] == "plugin"
    assert m["plugin"] is True
    assert m["managed"] is True
    assert m["api"] == "POST /v1/audio/speech"
    assert m["size_bytes"] == 32
    assert m["running"] is True


def test_plugin_models_empty_when_none_installed(monkeypatch):
    monkeypatch.setattr(plugins, "is_installed", lambda recipe: False)
    assert plugins.plugin_models() == []


# ------------------------------------------------------------------
# Çalışan eklenti portu (kategori köprüsü)
# ------------------------------------------------------------------

def test_running_port_for_category_prefers_order(monkeypatch):
    """Aynı kategoride birden çok eklenti çalışıyorsa prefer sırası uygulanır."""
    monkeypatch.setattr(plugins, "is_installed", lambda r: True)
    monkeypatch.setattr(plugins, "is_running", lambda r: r.id in {"muse_talk", "latent_sync"})
    monkeypatch.setattr(plugins, "_read_state", lambda r: {"port": 6000})

    assert plugins.running_port_for_category("lipsync", prefer=("latent_sync",))[0] == "latent_sync"
    assert plugins.running_port_for_category("lipsync")[0] == "muse_talk"


def test_running_port_for_category_none_when_stopped(monkeypatch):
    monkeypatch.setattr(plugins, "is_installed", lambda r: True)
    monkeypatch.setattr(plugins, "is_running", lambda r: False)
    monkeypatch.setattr(plugins, "_read_state", lambda r: {"port": 5000})

    assert plugins.running_port_for_category("upscale") is None


def test_running_port_for_category_requires_installed(monkeypatch):
    monkeypatch.setattr(plugins, "is_installed", lambda r: False)
    monkeypatch.setattr(plugins, "is_running", lambda r: True)
    monkeypatch.setattr(plugins, "_read_state", lambda r: {"port": 5000})

    assert plugins.running_port_for_category("upscale") is None




# ------------------------------------------------------------------
# /panel/plugins listeleme
# ------------------------------------------------------------------

def test_plugins_endpoint_requires_auth(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/panel/plugins")
    assert resp.status_code == 401


def test_plugins_endpoint_lists_catalog(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    resp = client.get("/panel/plugins")
    assert resp.status_code == 200
    data = resp.json()["plugins"]
    assert len(data) == 7
    assert all(p["installed"] is False for p in data)
    assert all(p["status"] == "available" for p in data)


# ------------------------------------------------------------------
# Kurulum
# ------------------------------------------------------------------

def test_install_unknown_plugin_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    resp = client.post("/panel/plugins/yok/install")
    assert resp.status_code == 404


def test_install_available_plugin_starts_install(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _install_noop(monkeypatch)
    resp = client.post("/panel/plugins/real_esrgan/install")
    assert resp.status_code == 200
    assert resp.json()["status"] == "installing"


def test_install_already_installed_409(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(plugins, "is_installed", lambda recipe: True)
    _login(client)
    resp = client.post("/panel/plugins/xtts_v2/install")
    assert resp.status_code == 409


def test_install_status_reflects_completion(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _install_noop(monkeypatch)
    client.post("/panel/plugins/real_esrgan/install")
    resp = client.get("/panel/plugins/real_esrgan/install/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is True
    assert body["installing"] is False


def test_install_status_unknown_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    resp = client.get("/panel/plugins/yok/install/status")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# Süreç denetimi (başlat / durdur / yeniden başlat) — taklit süreçle
# ------------------------------------------------------------------

def _fake_runtime(tmp_path, monkeypatch):
    """Başlatma için gereken ortamı (python, adaptör, port, Popen) taklit eder."""
    monkeypatch.setattr(plugins, "is_installed", lambda recipe: True)
    monkeypatch.setattr(plugins.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(plugins, "venv_python", lambda recipe: Path(sys.executable))
    monkeypatch.setattr(plugins, "_pick_port", lambda recipe: 19999)
    monkeypatch.setattr(plugins, "_terminate", lambda pid, recipe=None: None)


def test_start_not_installed_409(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    resp = client.post("/panel/plugins/real_esrgan/start")
    assert resp.status_code == 409


def test_start_installed_ok(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _fake_runtime(tmp_path, monkeypatch)
    resp = client.post("/panel/plugins/real_esrgan/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["port"] == 19999


def test_lifecycle_start_stop_restart(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _fake_runtime(tmp_path, monkeypatch)

    assert client.post("/panel/plugins/real_esrgan/start").json()["ok"] is True

    stop_resp = client.post("/panel/plugins/real_esrgan/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["running"] is False

    restart_resp = client.post("/panel/plugins/real_esrgan/restart")
    assert restart_resp.status_code == 200
    assert restart_resp.json()["ok"] is True


def test_stop_when_not_running_is_noop(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _fake_runtime(tmp_path, monkeypatch)
    resp = client.post("/panel/plugins/real_esrgan/stop")
    assert resp.status_code == 200
    assert resp.json()["already_stopped"] is True


# ------------------------------------------------------------------
# Log
# ------------------------------------------------------------------

def test_log_endpoint_returns_service_log(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    runtime = tmp_path / "runtime" / "plugins" / "real_esrgan"
    runtime.mkdir(parents=True)
    (runtime / "service.log").write_text("merhaba log", encoding="utf-8")
    (runtime / "install.log").write_text("kurulum logu", encoding="utf-8")

    service = client.get("/panel/plugins/real_esrgan/log?kind=service")
    assert service.status_code == 200
    assert service.json()["text"] == "merhaba log"

    install = client.get("/panel/plugins/real_esrgan/log?kind=install")
    assert install.json()["text"] == "kurulum logu"


def test_log_endpoint_missing_returns_empty(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    resp = client.get("/panel/plugins/real_esrgan/log")
    assert resp.status_code == 200
    assert resp.json()["text"] == ""


# ------------------------------------------------------------------
# Kaldırma
# ------------------------------------------------------------------

def test_remove_plugin(tmp_path, monkeypatch):
    """Kaldırma; runtime + venv + kaynak + ağırlıklar dahil her şeyi siler."""
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _fake_runtime(tmp_path, monkeypatch)
    runtime = tmp_path / "runtime" / "plugins" / "real_esrgan"
    venv = tmp_path / "venvs" / "real_esrgan"
    src = tmp_path / "src" / "real_esrgan"
    weights = tmp_path / "weights" / "real_esrgan"
    for directory in (runtime, venv, src, weights):
        directory.mkdir(parents=True)
        (directory / "marker.txt").write_text("x", encoding="utf-8")

    resp = client.delete("/panel/plugins/real_esrgan")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    for directory in (runtime, venv, src, weights):
        assert not directory.exists(), directory


def test_remove_unknown_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    resp = client.delete("/panel/plugins/yok")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# Sağlamlaştırma: disk/VRAM denetimi, kalıntı, yabancı pid, kilitler
# ------------------------------------------------------------------

def test_disk_check_reports_insufficient(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    monkeypatch.setattr(plugins, "_free_disk_mb", lambda path: 10.0)
    check = plugins.disk_check(recipe)
    assert check["ok"] is False
    assert check["required_mb"] >= recipe.disk_mb


def test_disk_check_reports_sufficient(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    monkeypatch.setattr(plugins, "_free_disk_mb", lambda path: 10_000_000.0)
    assert plugins.disk_check(recipe)["ok"] is True


def test_install_rejects_low_disk_and_releases_mutex(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    monkeypatch.setattr(
        plugins, "disk_check",
        lambda recipe, margin_mb=1024: {"ok": False, "free_mb": 5, "required_mb": 4024},
    )
    with pytest.raises(plugins.PluginError):
        plugins.install(recipe)
    # Başarısız kurulum kilidi bırakmalı (sonraki kurulum engellenmesin).
    assert plugins._INSTALL_MUTEX.acquire(blocking=False) is True
    plugins._INSTALL_MUTEX.release()


def test_install_blocked_when_another_running(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    assert plugins._INSTALL_MUTEX.acquire(blocking=False) is True
    try:
        with pytest.raises(plugins.PluginError):
            plugins.install(recipe)
    finally:
        plugins._INSTALL_MUTEX.release()


def test_remove_refuses_during_install(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._INSTALLS[recipe.id] = {"running": True}
    try:
        with pytest.raises(plugins.PluginError):
            plugins.remove(recipe)
    finally:
        plugins._INSTALLS.pop(recipe.id, None)


def test_vram_check(monkeypatch):
    import types
    from app import gpu_detect

    monkeypatch.setattr(
        gpu_detect, "detect_hardware",
        lambda: types.SimpleNamespace(vram_total_mb=8000),
    )
    assert plugins.vram_check(plugins.get_plugin("seedvr2_3b"))["ok"] is False
    assert plugins.vram_check(plugins.get_plugin("real_esrgan"))["ok"] is True

    monkeypatch.setattr(
        gpu_detect, "detect_hardware",
        lambda: types.SimpleNamespace(vram_total_mb=0),
    )
    assert plugins.vram_check(plugins.get_plugin("seedvr2_3b"))["ok"] is True


def test_artifacts_present_and_size(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    assert _ARTIFACTS_PRESENT_REAL(recipe) is False
    venv = plugins.venv_dir(recipe)
    venv.mkdir(parents=True)
    (venv / "parca.bin").write_bytes(b"x" * (2 * 1024 * 1024))  # ~2 MB
    assert _ARTIFACTS_PRESENT_REAL(recipe) is True
    assert _ARTIFACTS_SIZE_REAL(recipe) >= 1.0


def test_leftovers_status_and_size_in_panel(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    monkeypatch.setattr(plugins, "artifacts_present", lambda recipe: True)
    monkeypatch.setattr(plugins, "artifacts_size_mb", lambda recipe: 12.5)
    data = {p["id"]: p for p in client.get("/panel/plugins").json()["plugins"]}
    assert data["real_esrgan"]["has_leftovers"] is True
    assert data["real_esrgan"]["status"] == "leftovers"
    assert data["real_esrgan"]["artifacts_mb"] == 12.5


def test_stale_state_reconciled(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._write_state(recipe, pid=999999, port=1234, running=True)
    data = plugins._plugin_dict(recipe)
    assert data["running"] is False
    assert data["port"] == 0
    assert plugins._read_state(recipe)["running"] is False


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="yalnızca Linux /proc")
def test_pid_is_this_plugin_rejects_foreign_pid(tmp_path, monkeypatch):
    import os

    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    # Kendi sürecimiz bu eklentinin adaptörünü çalıştırmıyor.
    assert plugins._pid_is_this_plugin(os.getpid(), recipe) is False
    # Var olmayan pid de reddedilir.
    assert plugins._pid_is_this_plugin(2_147_483_600, recipe) is False


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="yalnızca Linux /proc")
def test_stop_foreign_pid_not_terminated(tmp_path, monkeypatch):
    import os

    client = _client(tmp_path, monkeypatch)
    _login(client)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._write_state(recipe, pid=os.getpid(), port=1, running=True)
    calls = []
    monkeypatch.setattr(plugins, "_terminate", lambda pid, recipe=None: calls.append(pid))
    resp = client.post("/panel/plugins/real_esrgan/stop")
    assert resp.status_code == 200
    assert resp.json().get("foreign_pid") is True
    assert calls == []
    assert plugins._read_state(recipe)["running"] is False


def test_is_installed_requires_venv(tmp_path, monkeypatch):
    """Durum 'kurulu' dese de venv yoksa kurulu sayılmaz (bayat durum)."""
    monkeypatch.setattr(plugins, "PLUGINS_RUNTIME_DIR", tmp_path / "runtime" / "plugins")
    monkeypatch.setattr(plugins, "venv_dir", lambda recipe: tmp_path / "venvs" / recipe.id)
    monkeypatch.setattr(plugins, "src_dir", lambda recipe: tmp_path / "src" / recipe.id)
    monkeypatch.setattr(plugins, "weights_dir", lambda recipe: tmp_path / "w" / recipe.id)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._write_state(recipe, installed=True)

    assert plugins.is_installed(recipe) is False  # venv yok
    (tmp_path / "venvs" / recipe.id / "bin").mkdir(parents=True)
    (tmp_path / "venvs" / recipe.id / "bin" / "python").write_text("x", encoding="utf-8")
    assert plugins.is_installed(recipe) is True


# ------------------------------------------------------------------
# Kurulum iptali ve dosyalarla birlikte kaldırma
# ------------------------------------------------------------------

def test_cancel_install_marks_flag_and_terminates_proc(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._INSTALLS[recipe.id] = {"running": True, "step": "ağırlıklar"}
    plugins._INSTALL_PROCS[recipe.id] = object()
    terminated = []
    monkeypatch.setattr(plugins, "_terminate_proc", lambda proc, grace=3.0: terminated.append(proc))
    try:
        assert plugins.cancel_install(recipe) is True
        assert recipe.id in plugins._INSTALL_CANCEL
        assert len(terminated) == 1
        assert plugins.install_status(recipe.id)["step"] == "iptal ediliyor"
        assert plugins.cancel_install(recipe) is True  # ikinci istek de kabul edilir
        assert plugins.cancel_install(plugins.get_plugin("latent_sync")) is False
    finally:
        plugins._INSTALLS.pop(recipe.id, None)
        plugins._INSTALL_PROCS.pop(recipe.id, None)
        plugins._INSTALL_CANCEL.discard(recipe.id)


def test_cancel_install_endpoint(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._INSTALLS[recipe.id] = {"running": True, "step": "Python ortamı"}
    monkeypatch.setattr(plugins, "_terminate_proc", lambda proc, grace=3.0: None)
    try:
        resp = client.post("/panel/plugins/real_esrgan/install/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelling"
        assert recipe.id in plugins._INSTALL_CANCEL
    finally:
        plugins._INSTALLS.pop(recipe.id, None)
        plugins._INSTALL_CANCEL.discard(recipe.id)

    resp = client.post("/panel/plugins/real_esrgan/install/cancel")
    assert resp.status_code == 409


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="yalnızca Linux /proc")
def test_run_cancellation_terminates_process(tmp_path, monkeypatch):
    import threading
    import time

    _client(tmp_path, monkeypatch)
    log_path = tmp_path / "run.log"
    log = log_path.open("w", encoding="utf-8")
    errors = []

    def worker():
        try:
            plugins._run(["sh", "-c", "sleep 30"], log, plugin_id="iptal-test")
        except plugins.PluginInstallCancelled as exc:  # pragma: no branch - beklenir
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            with plugins._INSTALL_CANCEL_LOCK:
                if "iptal-test" in plugins._INSTALL_PROCS:
                    break
            time.sleep(0.02)
        with plugins._INSTALL_CANCEL_LOCK:
            assert "iptal-test" in plugins._INSTALL_PROCS
            plugins._INSTALL_CANCEL.add("iptal-test")
        thread.join(timeout=10)
        assert thread.is_alive() is False
        assert errors
    finally:
        with plugins._INSTALL_CANCEL_LOCK:
            plugins._INSTALL_CANCEL.discard("iptal-test")
            plugins._INSTALL_PROCS.pop("iptal-test", None)
        log.close()


def test_install_status_exposes_progress_fields(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _login(client)
    recipe = plugins.get_plugin("real_esrgan")
    plugins._INSTALLS[recipe.id] = {
        "running": True, "done": False, "cancelled": False, "step": "ağırlıklar",
        "step_index": 3, "step_total": 5, "received": 512, "total": 1024, "pct": 50.0,
    }
    try:
        body = client.get("/panel/plugins/real_esrgan/install/status").json()
        assert body["installing"] is True
        assert body["step"] == "ağırlıklar"
        assert body["step_index"] == 3 and body["step_total"] == 5
        assert body["received"] == 512 and body["total"] == 1024
        assert body["pct"] == 50.0

        card = {p["id"]: p for p in client.get("/panel/plugins").json()["plugins"]}
        assert card["real_esrgan"]["install"]["pct"] == 50.0
        assert card["real_esrgan"]["installing"] is True
    finally:
        plugins._INSTALLS.pop(recipe.id, None)


def test_remove_deletes_all_artifact_targets(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    runtime = tmp_path / "runtime" / "plugins" / recipe.id
    weights = tmp_path / "weights" / recipe.id
    runtime.mkdir(parents=True)
    weights.mkdir(parents=True)
    (runtime / "state.json").write_text("{}", encoding="utf-8")
    (weights / "model.bin").write_bytes(b"x" * 1024)
    monkeypatch.setattr(plugins, "_artifact_paths", lambda r: [runtime, weights])

    result = plugins.remove(recipe)

    assert result["ok"] is True
    assert result["failed"] == []
    assert not runtime.exists() and not weights.exists()
    assert result["removed"]


def test_remove_reports_targets_that_could_not_be_deleted(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    recipe = plugins.get_plugin("real_esrgan")
    stubborn = tmp_path / "stubborn"
    stubborn.mkdir()
    monkeypatch.setattr(plugins, "_artifact_paths", lambda r: [stubborn])
    monkeypatch.setattr(plugins, "_force_rmtree", lambda path: None)

    result = plugins.remove(recipe)

    assert result["ok"] is False
    assert result["failed"] == [str(stubborn)]
    assert stubborn.exists()
