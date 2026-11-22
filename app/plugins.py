# ─────────────────────────────────────────────────────────────
#  Bölüm:    Eklenti (Plugin) Yöneticisi
#  Dosya:    app/plugins.py
#  Amaç:     Panelde "Eklentiler" sayfasında listelenen harici
#            teknolojilerin (ses klonlama, dudak senkronu, görsel
#            büyütme) kurulum, kaldırma ve süreç (başlat/durdur)
#            yaşam döngüsünü tek noktadan yönetir.
#  Mekanik:  - CATALOG: kod içine gömülü eklenti tarifleri (repo, Python
#              sürümü, bağımlılıklar, ağırlıklar, servis adaptörü).
#            - Kurulum arka planda bir iş parçacığında çalışır; adımlar
#              ve çıktı runtime/plugins/<id>/install.log'a yazılır.
#            - Her eklenti kendi izole venv'inde bir yerel HTTP servisi
#              olarak başlatılır (plugins/adapters/*). Süreç pid'i
#              runtime/plugins/<id>/state.json'da saklanır.
#  Kullanım: from app import plugins; plugins.list_plugins()
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger("vprovider")

# Proje kökü ve çalışma klasörleri
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_RUNTIME_DIR = PROJECT_ROOT / "runtime" / "plugins"
ADAPTERS_DIR = PROJECT_ROOT / "plugins" / "adapters"
UV_DIR = PROJECT_ROOT / "runtime" / "uv"
UV_BIN = UV_DIR / "uv"

# Servis portları bu tabandan başlar (eklenti sırasına göre artar)
PORT_BASE = 19300

# Kurulum adımlarının durum takibi (bellek içi)
_INSTALLS: dict[str, dict] = {}
_INSTALL_LOCK = threading.Lock()

# Kurulum iptali: işaretlenen eklentinin süreç grubu sonlandırılır ve
# uzun süren adımlar (ağırlık indirme) her parçada bu kümeyi denetler.
_INSTALL_CANCEL: set[str] = set()
_INSTALL_PROCS: dict[str, subprocess.Popen] = {}
_INSTALL_CANCEL_LOCK = threading.Lock()

# Aynı anda tek eklenti kurulumu (disk/GPU paylaşımı) ve yaşam döngüsü serileştirmesi
_INSTALL_MUTEX = threading.Lock()
_LIFECYCLE_LOCK = threading.RLock()

# Çalışan süreç tutamakları (bu sunucu oturumu başlattığı süreçler)
_PROCS: dict[str, subprocess.Popen] = {}

# Bellek modu: keep = model GPU'da kalır, dynamic = boşta GPU bırakılır.
MEMORY_KEEP = "keep"
MEMORY_DYNAMIC = "dynamic"
_MEMORY_MODES = {MEMORY_KEEP, MEMORY_DYNAMIC}

# Eklenti servis gözcüsü (kurulu servisler her zaman ayakta kalsın)
_SUPERVISOR_STOP = threading.Event()
_SUPERVISOR_THREAD: threading.Thread | None = None


class PluginError(RuntimeError):
    """Eklenti yaşam döngüsü hatası."""


class PluginInstallCancelled(PluginError):
    """Kullanıcı eklenti kurulumunu iptal etti."""


# ------------------------------------------------------------------
# Tarifler
# ------------------------------------------------------------------

@dataclass(frozen=True)
class WeightSpec:
    """İndirilecek bir ağırlık deposu (HuggingFace)."""

    repo_id: str
    dest: str = ""                       # proje köküne göreli hedef klasör
    allow_patterns: tuple[str, ...] = ()  # boş = tüm dosyalar


@dataclass(frozen=True)
class PluginRecipe:
    """Tek bir eklentinin kurulum ve çalıştırma tarifi."""

    id: str
    name: str
    category: str                    # tts | lipsync | upscale
    repo: str                        # kullanıcıya gösterilen kaynak bağlantısı
    description: str
    python_version: str
    api: str
    # Kurulum
    git_url: str = ""
    git_ref: str = ""
    setup_script: str = ""           # varsa genel venv/pip adımları yerine çalışır
    pip_index_url: str = ""          # torch gibi özel indeksler için
    pip_packages: tuple[str, ...] = ()
    requirements_file: str = ""
    setup_commands: tuple[str, ...] = ()
    weights: tuple[WeightSpec, ...] = ()
    # Çalıştırma
    adapter: str = ""                # plugins/adapters/ altındaki betik
    adapter_args: tuple[str, ...] = ()
    health_path: str = "/health"
    # Yol geçersiz kılmaları (proje köküne göreli)
    venv_dir: str = ""
    src_dir: str = ""
    weights_dir: str = ""
    # Bilgi
    vram_mb: int = 0
    disk_mb: int = 0


CATALOG: tuple[PluginRecipe, ...] = (
    PluginRecipe(
        id="xtts_v2",
        name="coqui/XTTS-v2",
        category="tts",
        repo="https://huggingface.co/coqui/XTTS-v2",
        description="Çok dilli ses klonlama (TTS). Kısa bir referans sesten konuşma üretir.",
        python_version="3.14",
        api="POST /v1/audio/speech",
        setup_script="scripts/install-xtts.sh",
        weights=(WeightSpec("coqui/XTTS-v2", dest="models/tts/coqui_XTTS-v2"),),
        adapter="tts_server.py",
        # Kendi venv'inde kurulur (paylaşımlı runtime/qwen-tts/venv'den ayrık),
        # böylece bu eklentiyi kaldırmak qwen-tts venv'ine dokunmaz.
        venv_dir="",
        weights_dir="models/tts/coqui_XTTS-v2",
        vram_mb=2000,
        disk_mb=5000,
    ),
    PluginRecipe(
        id="muse_talk",
        name="TMElyralab/MuseTalk",
        category="lipsync",
        repo="https://github.com/TMElyralab/MuseTalk",
        description="Gerçek zamanlı dudak senkronu; fotoğraf + ses ile konuşan kafa videosu üretir.",
        python_version="3.10",
        api="POST /v1/videos/lipsync",
        git_url="https://github.com/TMElyralab/MuseTalk",
        pip_index_url="https://download.pytorch.org/whl/cu118",
        pip_packages=("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.1"),
        requirements_file="requirements.txt",
        weights=(
            WeightSpec("TMElyralab/MuseTalk", dest="models/lipsync/muse_talk"),
            WeightSpec("stabilityai/sd-vae-ft-mse", dest="models/lipsync/muse_talk/sd-vae-ft-mse"),
            WeightSpec("openai/whisper-tiny", dest="models/lipsync/muse_talk/whisper"),
        ),
        adapter="lipsync_server.py",
        adapter_args=("--engine", "muse_talk"),
        weights_dir="models/lipsync/muse_talk",
        vram_mb=7000,
        disk_mb=12000,
    ),
    PluginRecipe(
        id="latent_sync",
        name="ByteDance/LatentSync",
        category="lipsync",
        repo="https://github.com/bytedance/LatentSync",
        description="Difüzyon tabanlı dudak senkronu; videoyu verilen sese göre yeniden üretir.",
        python_version="3.10",
        api="POST /v1/videos/lipsync",
        git_url="https://github.com/bytedance/LatentSync",
        requirements_file="requirements.txt",
        weights=(
            WeightSpec("ByteDance/LatentSync-1.5", dest="models/lipsync/latent_sync/checkpoints"),
        ),
        adapter="lipsync_server.py",
        adapter_args=("--engine", "latent_sync"),
        weights_dir="models/lipsync/latent_sync",
        vram_mb=6000,
        disk_mb=8000,
    ),
    PluginRecipe(
        id="real_esrgan",
        name="ai-forever/Real-ESRGAN",
        category="upscale",
        repo="https://github.com/ai-forever/Real-ESRGAN",
        description="Görsel büyütme (super-resolution); görüntüleri 2x/4x netleştirir.",
        python_version="3.11",
        api="POST /v1/images/upscale",
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_packages=("torch", "torchvision"),
        setup_commands=(
            "pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git opencv-python pillow numpy",
        ),
        adapter="esrgan_server.py",
        adapter_args=("--scale", "4"),
        weights_dir="models/upscale/real_esrgan",
        vram_mb=2000,
        disk_mb=3000,
    ),
    PluginRecipe(
        id="seedvr2_3b",
        name="ByteDance-Seed/SeedVR2-3B",
        category="restore",
        repo="https://huggingface.co/ByteDance-Seed/SeedVR2-3B",
        description="Tek adımlı difüzyon ile görsel/video onarımı ve büyütme (video restoration).",
        python_version="3.10",
        api="POST /v1/videos/restore",
        git_url="https://github.com/ByteDance-Seed/SeedVR",
        requirements_file="requirements.txt",
        setup_commands=(
            "pip install flash_attn==2.5.9.post1 --no-build-isolation",
        ),
        weights=(
            WeightSpec("ByteDance-Seed/SeedVR2-3B", dest="models/restore/seedvr2_3b"),
        ),
        adapter="seedvr2_server.py",
        weights_dir="models/restore/seedvr2_3b",
        vram_mb=80000,
        disk_mb=16000,
    ),
    PluginRecipe(
        id="musicgen_small",
        name="facebook/musicgen-small",
        category="music",
        repo="https://huggingface.co/facebook/musicgen-small",
        description="Metinden müzik üretimi (text-to-music). Kısa enstrümantal parçalar üretir.",
        python_version="3.11",
        api="POST /v1/audio/music/generate",
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_packages=("torch", "torchvision"),
        setup_commands=(
            "pip install transformers scipy soundfile",
        ),
        weights=(WeightSpec("facebook/musicgen-small", dest="models/music/musicgen_small"),),
        adapter="musicgen_server.py",
        weights_dir="models/music/musicgen_small",
        vram_mb=3000,
        disk_mb=4000,
    ),
    PluginRecipe(
        id="stable_audio_open",
        name="stabilityai/stable-audio-open-1.0",
        category="music",
        repo="https://huggingface.co/stabilityai/stable-audio-open-1.0",
        description="Metinden ses/müzik üretimi (text-to-audio). 44.1 kHz stereo, 47 sn'ye kadar.",
        python_version="3.10",
        api="POST /v1/audio/music/generate",
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_packages=("torch", "torchaudio"),
        setup_commands=(
            "pip install stable-audio-tools einops",
        ),
        weights=(
            WeightSpec("stabilityai/stable-audio-open-1.0", dest="models/music/stable_audio_open"),
        ),
        adapter="stable_audio_server.py",
        weights_dir="models/music/stable_audio_open",
        vram_mb=4000,
        disk_mb=7000,
    ),
)


def get_recipe(plugin_id: str) -> PluginRecipe | None:
    """Kimliğe göre tarifi bulur; yoksa None."""
    return next((r for r in CATALOG if r.id == plugin_id), None)


def _project_path(rel: str) -> Path:
    return PROJECT_ROOT / rel


# ------------------------------------------------------------------
# Yol yardımcıları
# ------------------------------------------------------------------

def runtime_dir(recipe: PluginRecipe) -> Path:
    return PLUGINS_RUNTIME_DIR / recipe.id


def src_dir(recipe: PluginRecipe) -> Path:
    return _project_path(recipe.src_dir) if recipe.src_dir else runtime_dir(recipe) / "src"


def venv_dir(recipe: PluginRecipe) -> Path:
    return _project_path(recipe.venv_dir) if recipe.venv_dir else runtime_dir(recipe) / "venv"


def weights_dir(recipe: PluginRecipe) -> Path:
    if recipe.weights_dir:
        return _project_path(recipe.weights_dir)
    return _project_path(f"models/{recipe.category}/{recipe.id}")


def venv_python(recipe: PluginRecipe) -> Path:
    return venv_dir(recipe) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _state_file(recipe: PluginRecipe) -> Path:
    return runtime_dir(recipe) / "state.json"


def _install_log(recipe: PluginRecipe) -> Path:
    return runtime_dir(recipe) / "install.log"


def _service_log(recipe: PluginRecipe) -> Path:
    return runtime_dir(recipe) / "service.log"


# ------------------------------------------------------------------
# Durum okuma/yazma
# ------------------------------------------------------------------

def _read_state(recipe: PluginRecipe) -> dict:
    try:
        return json.loads(_state_file(recipe).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(recipe: PluginRecipe, **fields) -> dict:
    runtime_dir(recipe).mkdir(parents=True, exist_ok=True)
    state = _read_state(recipe)
    state.update(fields)
    _state_file(recipe).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def memory_mode(recipe: PluginRecipe) -> str:
    """Eklentinin bellek modunu döner (varsayılan: keep).

    keep    -> model istekler arasında GPU'da kalır (servis hep hazır).
    dynamic -> servis ayakta kalır, model boşta GPU'dan bırakılır.
    """
    mode = _read_state(recipe).get("memory_mode")
    return mode if mode in _MEMORY_MODES else MEMORY_KEEP


def _state_int(state: dict, key: str, default: int) -> int:
    try:
        return int(float(state.get(key, default)))
    except (TypeError, ValueError):
        return default


def set_memory_mode(recipe: PluginRecipe, mode: str) -> str:
    """Modu kaydeder ve (değiştiyse) servisi yeni boşta-bırakma ayarıyla başlatır."""
    if mode not in _MEMORY_MODES:
        raise PluginError("memory_mode yalnızca 'keep' veya 'dynamic' olabilir.")
    current = memory_mode(recipe)
    _write_state(recipe, memory_mode=mode)
    if current == mode:
        return mode
    with _LIFECYCLE_LOCK:
        if not is_installed(recipe) or not recipe.adapter:
            return mode
        try:
            if is_running(recipe):
                stop(recipe)
                time.sleep(0.5)
            start(recipe)
        except Exception:  # noqa: BLE001 - mod kaydı servis hatasına takılmasın
            logger.exception("Eklenti bellek modu uygulanamadı: %s", recipe.id)
    return mode


def _xtts_weights_present(recipe: PluginRecipe) -> bool:
    """Coqui XTTS ağırlıkları diskte var mı (config.json + model dosyası)?"""
    wd = weights_dir(recipe)
    if not (wd / "config.json").is_file():
        return False
    return any(wd.glob("*.pth")) or any(wd.glob("*.safetensors"))


def is_installed(recipe: PluginRecipe) -> bool:
    """Eklenti kurulu mu? (durum dosyası + gerçek venv/varlık denetimi)

    Durum dosyası 'kurulu' dese bile venv kaybolmuşsa (ör. venv yolu
    değişince) kurulu sayılmaz; bu hatalı 'kurulu görünüyor ama başlatılamıyor'
    durumunu önler. xtts_v2 için durum dosyası olmasa da ağırlık+venv
    varlığı yeterlidir.
    """
    state_installed = bool(_read_state(recipe).get("installed"))
    if recipe.id == "xtts_v2":
        # Durum dosyası olmasa da ağırlık+venv varlığı kurulu sayılır.
        return _xtts_weights_present(recipe) and venv_python(recipe).is_file()
    if not state_installed:
        return False
    if recipe.adapter and not venv_python(recipe).is_file():
        return False
    return True


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_running(recipe: PluginRecipe) -> bool:
    """Eklenti servisi çalışıyor mu?"""
    proc = _PROCS.get(recipe.id)
    if proc is not None:
        if proc.poll() is None:
            return True
        return False
    pid = _read_state(recipe).get("pid")
    if _pid_alive(pid):
        return True
    return False


# ------------------------------------------------------------------
# Disk / VRAM / kalıntı denetimleri
# ------------------------------------------------------------------

def _dir_size_mb(path: Path) -> float:
    """Bir dizinin toplam boyutu (MB); yoksa 0. Hatalı erişimler atlanır."""
    if not path.exists():
        return 0.0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return round(total / (1024 * 1024), 1)


def _free_disk_mb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except OSError:
        return float("inf")


def _artifact_paths(recipe: PluginRecipe) -> list[Path]:
    """Eklentiye bağlı tüm disk hedeflerini (iç içe olanları ayıklayarak) döner."""
    raw = [runtime_dir(recipe), venv_dir(recipe), src_dir(recipe), weights_dir(recipe)]
    raw += [_project_path(spec.dest) for spec in recipe.weights if spec.dest]
    unique: list[Path] = []
    for path in raw:
        if path not in unique:
            unique.append(path)
    # Başka bir hedefin altında kalan yolları at (çift sayımı önler).
    return [p for p in unique if not any(p != q and q in p.parents for q in unique)]


def artifacts_present(recipe: PluginRecipe) -> bool:
    """Kurulu olmasa bile diskte kalıntı var mı (yarım kalan kurulum)?"""
    return any(path.exists() for path in _artifact_paths(recipe))


def artifacts_size_mb(recipe: PluginRecipe) -> float:
    """Eklentiye bağlı kalıntıların toplam boyutu (MB)."""
    return round(sum(_dir_size_mb(path) for path in _artifact_paths(recipe)), 1)


def disk_check(recipe: PluginRecipe, margin_mb: int = 1024) -> dict:
    """Kurulum için yeterli boş disk var mı?

    Gerekli alan = tahmini disk_mb + emniyet payı (varsayılan 1 GB). Tahmin
    yoksa (0) en az 1.5 GB varsayılır. Yetersizse 'ok': False döner.
    """
    required = (recipe.disk_mb or 1500) + margin_mb
    free = _free_disk_mb(PROJECT_ROOT)
    return {"ok": free >= required, "free_mb": int(free), "required_mb": int(required)}


def vram_check(recipe: PluginRecipe) -> dict:
    """Eklentinin VRAM ihtiyacı toplam VRAM'e sığıyor mu (bilgi amaçlı)?

    Bilinmeyen VRAM (0) ya da ihtiyaç belirtilmemişse 'ok': True döner;
    amaç kurulumu engellemek değil, panelde uyarmaktır.
    """
    from app.gpu_detect import detect_hardware

    total = detect_hardware().vram_total_mb
    if recipe.vram_mb <= 0 or total <= 0:
        return {"ok": True, "total_mb": total, "required_mb": recipe.vram_mb}
    return {"ok": recipe.vram_mb <= total, "total_mb": total, "required_mb": recipe.vram_mb}


def _reconcile_state(recipe: PluginRecipe, running: bool) -> dict:
    """Bayat durumu düzeltir: state 'running' dese de süreç ölüyse temizler."""
    state = _read_state(recipe)
    if state.get("running") and not running:
        state = _write_state(recipe, running=False, pid=0)
    return state


def _plugin_dict(recipe: PluginRecipe) -> dict:
    installed = is_installed(recipe)
    running = is_running(recipe) if installed else False
    state = _reconcile_state(recipe, running)
    with _INSTALL_LOCK:
        install = dict(_INSTALLS.get(recipe.id) or {})
    leftovers = (not installed) and artifacts_present(recipe)
    return {
        "id": recipe.id,
        "name": recipe.name,
        "category": recipe.category,
        "repo": recipe.repo,
        "description": recipe.description,
        "python_version": recipe.python_version,
        "api": recipe.api,
        "vram_mb": recipe.vram_mb,
        "disk_mb": recipe.disk_mb,
        "installed": installed,
        "running": running,
        "memory_mode": memory_mode(recipe),
        "has_service": bool(recipe.adapter),
        "port": state.get("port", 0) if running else 0,
        "installing": bool(install.get("running")),
        "install": install,
        "cancelled": bool(install.get("cancelled")),
        "has_leftovers": leftovers,
        "artifacts_mb": artifacts_size_mb(recipe) if leftovers else 0.0,
        "vram_ok": vram_check(recipe)["ok"],
        "status": (
            "installing" if install and install.get("running")
            else "running" if running
            else "installed" if installed
            else "leftovers" if leftovers
            else "available"
        ),
    }


def list_plugins() -> list[dict]:
    """Katalogdaki tüm eklentileri canlı durumlarıyla döner."""
    return [_plugin_dict(recipe) for recipe in CATALOG]


def plugin_models() -> list[dict]:
    """Kurulu eklentileri panel "model" kayıtlarına çevirir.

    Modeller sayfasında eklentiyle gelen modellerin (TTS/dudak senkronu/
    büyütme/onarım/müzik) listelenip kategori filtresine girmesi için her
    KURULU eklenti tek bir model kartı olarak döner.
    """
    entries: list[dict] = []
    for recipe in CATALOG:
        if not is_installed(recipe):
            continue
        wd = weights_dir(recipe)
        size = 0
        if wd.is_dir():
            size = sum(p.stat().st_size for p in wd.rglob("*") if p.is_file())
        running = is_running(recipe)
        entries.append({
            "id": recipe.id,
            "name": recipe.name,
            "path": str(wd),
            "size_bytes": size,
            "category": recipe.category,
            "kind": "plugin",
            "loaded": running,
            "memory_mode": memory_mode(recipe),
            "params": "",
            "managed": True,
            "plugin": True,
            "service_running": running,
            "api": recipe.api,
            "vram_mb": recipe.vram_mb,
            "running": running,
        })
    return entries


def get_plugin(plugin_id: str) -> PluginRecipe | None:
    return get_recipe(plugin_id)


# ------------------------------------------------------------------
# uv (Python sürüm yöneticisi) yüklemesi
# ------------------------------------------------------------------

def _uv_binary() -> Path | None:
    if UV_BIN.is_file():
        return UV_BIN
    found = shutil.which("uv")
    return Path(found) if found else None


def _ensure_uv(log, plugin_id: str | None = None) -> Path:
    """uv'yi bulur; yoksa resmi kurulum betiğiyle runtime/uv içine kurar."""
    existing = _uv_binary()
    if existing:
        return existing
    log.write(" • uv bulunamadı; runtime/uv içine kuruluyor...\n")
    log.flush()
    uv_dir = UV_DIR
    uv_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_INSTALL_DIR"] = str(uv_dir)
    env["UV_NO_MODIFY_PATH"] = "1"
    _run_shell(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        log,
        env=env,
        plugin_id=plugin_id,
    )
    if UV_BIN.is_file():
        return UV_BIN
    found = _uv_binary()
    if found:
        return found
    raise PluginError("uv kurulumu başarısız oldu.")


# ------------------------------------------------------------------
# Alt süreç yardımcıları
# ------------------------------------------------------------------

def _is_install_cancelled(plugin_id: str | None) -> bool:
    """Eklenti kurulumu için iptal işareti konmuş mu?"""
    if not plugin_id:
        return False
    with _INSTALL_CANCEL_LOCK:
        return plugin_id in _INSTALL_CANCEL


def _terminate_proc(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Süreç grubunu nazikçe, olmazsa zorla sonlandırır."""
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:  # pragma: no cover - Windows yolu
            proc.terminate()
    except (ProcessLookupError, OSError):
        return
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows yolu
            proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _run(
    cmd: list[str],
    log,
    cwd: Path | None = None,
    env: dict | None = None,
    plugin_id: str | None = None,
) -> None:
    """Komutu çalıştırır; çıktısını log'a akıtır; hata/iptalde yükseltir.

    plugin_id verilirse süreç kurulum kaydına alınır ve 0.2 sn'lik
    aralıklarla iptal işareti denetlenir; iptalde süreç grubu sonlandırılır.
    """
    if _is_install_cancelled(plugin_id):
        raise PluginInstallCancelled("Kurulum iptal edildi.")
    log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )
    if plugin_id:
        with _INSTALL_CANCEL_LOCK:
            _INSTALL_PROCS[plugin_id] = proc
    try:
        while True:
            try:
                returncode = proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if _is_install_cancelled(plugin_id):
                    _terminate_proc(proc)
                    raise PluginInstallCancelled("Kurulum iptal edildi.") from None
        if _is_install_cancelled(plugin_id):
            raise PluginInstallCancelled("Kurulum iptal edildi.")
        if returncode != 0:
            raise PluginError(
                f"Komut başarısız (kod {returncode}): {' '.join(str(c) for c in cmd)}"
            )
    finally:
        if plugin_id:
            with _INSTALL_CANCEL_LOCK:
                if _INSTALL_PROCS.get(plugin_id) is proc:
                    _INSTALL_PROCS.pop(plugin_id, None)


def _run_shell(
    command: str,
    log,
    cwd: Path | None = None,
    env: dict | None = None,
    plugin_id: str | None = None,
) -> None:
    _run(["sh", "-c", command], log, cwd=cwd, env=env, plugin_id=plugin_id)


def _venv_pip_env(recipe: PluginRecipe) -> dict:
    """venv/bin'i PATH başına ekleyen ortam (pip/mim çağrıları için)."""
    env = os.environ.copy()
    bin_dir = venv_dir(recipe) / ("Scripts" if os.name == "nt" else "bin")
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv_dir(recipe))
    return env


# ------------------------------------------------------------------
# Kurulum
# ------------------------------------------------------------------

def _set_install(plugin_id: str, **fields) -> None:
    with _INSTALL_LOCK:
        current = _INSTALLS.setdefault(plugin_id, {})
        current.update(fields)


def install_status(plugin_id: str) -> dict | None:
    with _INSTALL_LOCK:
        status = _INSTALLS.get(plugin_id)
        return dict(status) if status else None


def _install_step_total(recipe: PluginRecipe) -> int:
    """Kurulumda gösterilecek yaklaşık adım sayısı (ilerleme çubuğu için)."""
    total = 1  # hazırlık
    if recipe.setup_script:
        total += 1
    else:
        if recipe.git_url:
            total += 1
        total += 1  # Python ortamı
        if recipe.pip_packages:
            total += 1
        if recipe.requirements_file:
            total += 1
        total += len(recipe.setup_commands)
    if recipe.weights:
        total += 1
    total += 1  # tamamlandı
    return total


def _advance_step(recipe: PluginRecipe, step: str) -> None:
    """Kurulumu bir sonraki adıma geçirir; iptal işaretini denetler."""
    if _is_install_cancelled(recipe.id):
        raise PluginInstallCancelled("Kurulum iptal edildi.")
    with _INSTALL_LOCK:
        current = _INSTALLS.setdefault(recipe.id, {})
        current["step_index"] = int(current.get("step_index", 0)) + 1
        current["step"] = step
        current["updated_at"] = time.time()
        if not current.get("step_total"):
            current["step_total"] = _install_step_total(recipe)


def _do_install(recipe: PluginRecipe) -> None:
    runtime_dir(recipe).mkdir(parents=True, exist_ok=True)
    with open(_install_log(recipe), "a", encoding="utf-8") as log:
        log.write(f"\n=== {recipe.name} kurulumu başladı ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
        log.flush()
        try:
            _set_install(
                recipe.id,
                running=True,
                step="hazırlık",
                step_index=1,
                step_total=_install_step_total(recipe),
                received=0,
                total=0,
                pct=None,
                cancelled=False,
                error=None,
                done=False,
                updated_at=time.time(),
            )
            log.write(" • Çalışma klasörleri hazırlanıyor...\n")
            log.flush()
            venv_dir(recipe).parent.mkdir(parents=True, exist_ok=True)
            weights_dir(recipe).mkdir(parents=True, exist_ok=True)
            # Git klonu olmayan eklentilerde de setup_commands için kaynak
            # klasörü mevcut olmalı (pip/adapter burayı cwd kullanır).
            src_dir(recipe).mkdir(parents=True, exist_ok=True)

            if recipe.setup_script:
                _advance_step(recipe, "ortam")
                log.write(f" • Kurulum betiği çalıştırılıyor: {recipe.setup_script}\n")
                log.flush()
                _run_shell(
                    f"bash {recipe.setup_script}",
                    log,
                    cwd=PROJECT_ROOT,
                    plugin_id=recipe.id,
                )
            else:
                uv = _ensure_uv(log, plugin_id=recipe.id)

                if recipe.git_url:
                    _advance_step(recipe, "kaynak kodu")
                    log.write(f" • Kaynak kodu indiriliyor: {recipe.git_url}\n")
                    log.flush()
                    _clone_source(recipe, log)

                _advance_step(recipe, "Python ortamı")
                log.write(f" • Python {recipe.python_version} ortamı oluşturuluyor...\n")
                log.flush()
                _run(
                    [str(uv), "venv", "--python", recipe.python_version, str(venv_dir(recipe))],
                    log,
                    plugin_id=recipe.id,
                )

                if recipe.pip_packages:
                    _advance_step(recipe, "torch/çekirdek paketler")
                    cmd = [str(uv), "pip", "install", "--python", str(venv_python(recipe))]
                    if recipe.pip_index_url:
                        cmd += ["--index-url", recipe.pip_index_url]
                    cmd += list(recipe.pip_packages)
                    _run(cmd, log, plugin_id=recipe.id)

                req = src_dir(recipe) / recipe.requirements_file if recipe.requirements_file else None
                if req and req.is_file():
                    _advance_step(recipe, "bağımlılıklar")
                    log.write(f" • Bağımlılıklar kuruluyor: {recipe.requirements_file}\n")
                    log.flush()
                    _run(
                        [str(uv), "pip", "install", "--python", str(venv_python(recipe)), "-r", str(req)],
                        log,
                        plugin_id=recipe.id,
                    )

                for command in recipe.setup_commands:
                    _advance_step(recipe, "ek kurulum")
                    _run_shell(
                        command,
                        log,
                        cwd=src_dir(recipe),
                        env=_venv_pip_env(recipe),
                        plugin_id=recipe.id,
                    )

            if recipe.weights:
                _advance_step(recipe, "ağırlıklar")
                _download_weights(recipe, log)

            _advance_step(recipe, "tamamlandı")
            _set_install(
                recipe.id,
                running=False,
                done=True,
                cancelled=False,
                error=None,
                pct=100,
                step_index=_install_step_total(recipe),
                updated_at=time.time(),
            )
            _write_state(recipe, installed=True, installed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            log.write(" ✓ Kurulum tamamlandı.\n")
            log.flush()
            logger.info("Eklenti kuruldu: %s", recipe.id)
            if recipe.adapter:
                try:
                    start(recipe)
                    log.write(" ✓ Servis arka planda başlatıldı.\n")
                    log.flush()
                except Exception:  # noqa: BLE001 - kurulum yine de başarılı sayılır
                    logger.exception("Kurulan eklenti otomatik başlatılamadı: %s", recipe.id)
                    log.write(" ! Servis otomatik başlatılamadı; panelden deneyin.\n")
                    log.flush()
        except PluginInstallCancelled:
            logger.info("Eklenti kurulumu iptal edildi: %s", recipe.id)
            _set_install(
                recipe.id,
                running=False,
                done=True,
                cancelled=True,
                step="iptal",
                error=None,
                pct=None,
                updated_at=time.time(),
            )
            log.write(" ✗ Kurulum iptal edildi.\n")
            log.flush()
        except Exception as exc:  # noqa: BLE001 - arayüze her hatayı taşımak istiyoruz
            logger.exception("Eklenti kurulumu başarısız: %s", recipe.id)
            _set_install(
                recipe.id,
                running=False,
                done=True,
                cancelled=False,
                step="hata",
                error=str(exc),
                pct=None,
                updated_at=time.time(),
            )
            log.write(f" ✗ HATA: {exc}\n")
            log.flush()


def _clone_source(recipe: PluginRecipe, log) -> None:
    target = src_dir(recipe)
    if (target / ".git").is_dir():
        _run(["git", "-C", str(target), "fetch", "--depth", "1", "origin"], log, plugin_id=recipe.id)
        _run(["git", "-C", str(target), "reset", "--hard", "FETCH_HEAD"], log, plugin_id=recipe.id)
        return
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if recipe.git_ref:
        cmd += ["--branch", recipe.git_ref]
    cmd += [recipe.git_url, str(target)]
    _run(cmd, log, plugin_id=recipe.id)


def _download_weights(recipe: PluginRecipe, log) -> None:
    """Ağırlıkları toplam ilerlemeyle ve iptal edilebilir şekilde indirir.

    Her dosya Range destekli ``.part`` akışıyla iner; ilerleme tüm
    depolar için toplanır ve panel ``/install/status`` üzerinden izler.
    İptalde yalnızca yarım ``.part`` dosyaları silinir.
    """
    from app import hf_downloader

    token = settings.hf_token or None
    prepared: list[tuple[WeightSpec, dict[str, int] | None]] = []
    overall_total = 0
    for spec in recipe.weights:
        try:
            sizes = hf_downloader._repo_file_sizes(spec.repo_id, token=token)
        except Exception:  # noqa: BLE001 - boyut alınamazsa indirme yine denenir
            logger.warning("Ağırlık boyutları alınamadı: %s", spec.repo_id, exc_info=True)
            sizes = None
        if sizes:
            allow = list(spec.allow_patterns)
            overall_total += sum(
                size for name, size in sizes.items()
                if not allow or hf_downloader._matches_any(name, allow)
            )
        prepared.append((spec, sizes))

    done = 0
    for spec, sizes in prepared:
        if _is_install_cancelled(recipe.id):
            raise PluginInstallCancelled("Kurulum iptal edildi.")
        dest = _project_path(spec.dest) if spec.dest else weights_dir(recipe)
        dest.mkdir(parents=True, exist_ok=True)
        log.write(f" • Ağırlık indiriliyor: {spec.repo_id} → {dest}\n")
        log.flush()

        def _progress(got: int, file_total: int, base: int = done) -> None:
            aggregate = base + got
            denominator = max(overall_total, aggregate, file_total + base)
            pct = round(aggregate * 100 / denominator, 1) if denominator else None
            _set_install(
                recipe.id,
                received=aggregate,
                total=denominator,
                pct=pct,
                updated_at=time.time(),
            )

        try:
            result = hf_downloader.download_files(
                spec.repo_id,
                dest,
                allow_patterns=list(spec.allow_patterns) or None,
                token=token,
                progress_cb=_progress,
                cancel_check=lambda: _is_install_cancelled(recipe.id),
                sizes=sizes,
            )
        except hf_downloader.DownloadCancelledError:
            raise PluginInstallCancelled("Kurulum iptal edildi.") from None
        done += int(result["received"])

    denominator = max(overall_total, done)
    _set_install(
        recipe.id,
        received=done,
        total=denominator,
        pct=100 if denominator and done >= denominator else None,
        updated_at=time.time(),
    )


def install(recipe: PluginRecipe) -> None:
    """Kurulumu arka planda başlatır (zaten sürüyorsa yok sayar).

    Aynı anda yalnızca tek eklenti kurulabilir (disk/GPU payı). Kurulumdan
    önce boş disk alanı denetlenir; yetersizse net bir hata verilir.
    """
    if is_installed(recipe):
        raise PluginError(f"'{recipe.name}' zaten kurulu.")
    with _INSTALL_LOCK:
        status = _INSTALLS.get(recipe.id)
        if status and status.get("running"):
            raise PluginError(f"'{recipe.name}' kurulumu zaten sürüyor.")
    if not _INSTALL_MUTEX.acquire(blocking=False):
        raise PluginError("Başka bir eklenti kurulumu sürüyor; bitince tekrar deneyin.")
    try:
        check = disk_check(recipe)
        if not check["ok"]:
            raise PluginError(
                f"Yetersiz disk alanı: '{recipe.name}' için ~{check['required_mb']} MB "
                f"gerekli, boş alan yalnızca {check['free_mb']} MB."
            )
        # Önceki iptal işaretleri/tutamaklar yeni kurulumu etkilemesin.
        with _INSTALL_CANCEL_LOCK:
            _INSTALL_CANCEL.discard(recipe.id)
            _INSTALL_PROCS.pop(recipe.id, None)
        with _INSTALL_LOCK:
            _INSTALLS[recipe.id] = {
                "running": True,
                "step": "başlatılıyor",
                "step_index": 0,
                "step_total": _install_step_total(recipe),
                "received": 0,
                "total": 0,
                "pct": None,
                "cancelled": False,
                "error": None,
                "done": False,
                "updated_at": time.time(),
            }
        thread = threading.Thread(target=_do_install_guarded, args=(recipe,), daemon=True)
        thread.start()
    except Exception:
        _INSTALL_MUTEX.release()
        raise


def cancel_install(recipe: PluginRecipe) -> bool:
    """Süren eklenti kurulumunu iptal eder (süreç grubu sonlandırılır).

    İptal işareti konur; uzun süren indirme döngüsü her parçada bunu
    denetler. Alt süreç varsa nazikçe, gerekirse zorla kapatılır.
    Kurulum yoksa False döner.
    """
    with _INSTALL_LOCK:
        status = _INSTALLS.get(recipe.id)
        if not status or not status.get("running"):
            return False
        status["step"] = "iptal ediliyor"
        status["updated_at"] = time.time()
    with _INSTALL_CANCEL_LOCK:
        _INSTALL_CANCEL.add(recipe.id)
        proc = _INSTALL_PROCS.get(recipe.id)
    if proc is not None:
        _terminate_proc(proc)
    return True


def _do_install_guarded(recipe: PluginRecipe) -> None:
    """_do_install'i çalıştırır ve kurulum kilidini her durumda bırakır."""
    try:
        _do_install(recipe)
    finally:
        _INSTALL_MUTEX.release()


# ------------------------------------------------------------------
# Servis denetimi (başlat / durdur / yeniden başlat)
# ------------------------------------------------------------------

def _pick_port(recipe: PluginRecipe) -> int:
    index = CATALOG.index(recipe)
    for offset in range(50):
        port = PORT_BASE + index * 10 + offset
        if not _port_in_use(port):
            return port
    raise PluginError("Uygun boş port bulunamadı.")


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _idle_timeout_seconds(recipe: PluginRecipe) -> int:
    """Eklenti servisinin boşta GPU'yu bırakma süresi (saniye).

    keep modunda -1 döner (boşta bırakma kapalı). dynamic modda
    VPROVIDER_PLUGIN_IDLE_SECONDS ortam değişkeni kullanılır (varsayılan 120).
    """
    if memory_mode(recipe) == MEMORY_KEEP:
        return -1
    # İstek ile aktif (dynamic): model istek bitince anında boşaltılır.
    raw = os.environ.get("VPROVIDER_PLUGIN_IDLE_SECONDS", "0")
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def start(recipe: PluginRecipe) -> dict:
    """Eklenti servisini başlatır (yaşam döngüsü kilidi altında)."""
    with _LIFECYCLE_LOCK:
        if not is_installed(recipe):
            raise PluginError(f"'{recipe.name}' kurulu değil.")
        if not recipe.adapter:
            raise PluginError(f"'{recipe.name}' bir servis sunmuyor.")
        if is_running(recipe):
            return {"ok": True, "already_running": True, **_running_info(recipe)}

        python = venv_python(recipe)
        if not python.is_file():
            raise PluginError(f"Çalışma ortamı bulunamadı: {python}")

        port = _pick_port(recipe)
        adapter = ADAPTERS_DIR / recipe.adapter
        if not adapter.is_file():
            raise PluginError(f"Servis adaptörü bulunamadı: {adapter}")

        idle_timeout = _idle_timeout_seconds(recipe)
        cmd = [
            str(python),
            str(adapter),
            "--port", str(port),
            "--weights", str(weights_dir(recipe)),
            "--src", str(src_dir(recipe)),
            "--idle-timeout", str(idle_timeout),
            *recipe.adapter_args,
        ]
        log_path = _service_log(recipe)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write(f"\n=== servis başlatıldı ({time.strftime('%Y-%m-%d %H:%M:%S')}) port={port} ===\n")
        log_handle.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        _PROCS[recipe.id] = proc
        _write_state(
            recipe,
            pid=proc.pid,
            port=port,
            running=True,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            idle_timeout=idle_timeout,
            stopped_by_user=False,
        )
        logger.info("Eklenti başlatıldı: %s (pid=%s, port=%s)", recipe.id, proc.pid, port)
        time.sleep(0.5)
        if proc.poll() is not None:
            _write_state(recipe, running=False, pid=0)
            _PROCS.pop(recipe.id, None)
            raise PluginError(f"'{recipe.name}' servisi hemen kapandı; logu inceleyin.")
        return {"ok": True, "pid": proc.pid, "port": port, "running": True}


def stop(recipe: PluginRecipe) -> dict:
    """Eklenti servisini durdurur (yabancı pid'leri öldürmez)."""
    with _LIFECYCLE_LOCK:
        proc = _PROCS.pop(recipe.id, None)
        if proc is not None and proc.poll() is None:
            _terminate(proc.pid, recipe)
            _write_state(recipe, running=False, pid=0, stopped_by_user=True,
                         stopped_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("Eklenti durduruldu: %s (pid=%s)", recipe.id, proc.pid)
            return {"ok": True, "running": False}

        pid = _read_state(recipe).get("pid")
        if not pid:
            _write_state(recipe, running=False, pid=0, stopped_by_user=True)
            return {"ok": True, "running": False, "already_stopped": True}
        if not _pid_is_this_plugin(pid, recipe):
            logger.warning(
                "pid %s '%s' eklentisine ait görünmüyor; öldürülmedi.", pid, recipe.id
            )
            _write_state(recipe, running=False, pid=0, stopped_by_user=True)
            return {"ok": True, "running": False, "foreign_pid": True}
        _terminate(pid, recipe)
        _write_state(recipe, running=False, pid=0, stopped_by_user=True,
                     stopped_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Eklenti durduruldu: %s (pid=%s)", recipe.id, pid)
        return {"ok": True, "running": False}


def _pid_is_this_plugin(pid: int, recipe: PluginRecipe) -> bool:
    """pid gerçekten bu eklentinin adaptör süreci mi? (Linux /proc doğrulaması)

    /proc yoksa (Linux dışı) doğrulanamaz ve True döner; varsa cmdline içinde
    adaptör dosyası ya da venv yolu aranır. Böylece yeniden kullanılmış pid'in
    yanlış sürece SIGTERM göndermesi engellenir.
    """
    if not Path("/proc").is_dir():
        return True
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    return any(
        marker and marker in cmdline
        for marker in (recipe.adapter, str(venv_dir(recipe)))
    )


def _terminate(pid: int, recipe: PluginRecipe | None = None) -> None:
    if recipe is not None and not _pid_is_this_plugin(pid, recipe):
        logger.warning("pid %s '%s' eklentisine ait değil; sonlandırılmadı.", pid, recipe.id)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    for _ in range(30):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def restart(recipe: PluginRecipe) -> dict:
    """Servisi durdurup yeniden başlatır."""
    with _LIFECYCLE_LOCK:
        stop(recipe)
        time.sleep(0.5)
        return start(recipe)


_AUTO_START_COOLDOWN = 60.0
_AUTO_START_FAILED: dict[str, float] = {}


def start_installed() -> list[str]:
    """Kurulu eklenti servislerini arka planda hazır tutar.

    - Çalışmayan kurulu servisleri başlatır (kullanıcı açıkça durdurmadıysa).
    - Çalışan servisin boşta-bırakma ayarı modla uyuşmuyorsa yeniden başlatır.
    - Başarısız denemeleri bir süre tekrarlamaz (log/CPU spam'ini önler).
    """
    started: list[str] = []
    now = time.time()
    for recipe in CATALOG:
        if not recipe.adapter:
            continue
        try:
            if not is_installed(recipe):
                continue
            state = _read_state(recipe)
            expected = _idle_timeout_seconds(recipe)
            if is_running(recipe):
                if _state_int(state, "idle_timeout", -999) != expected:
                    logger.info("Eklenti boşta ayarı güncelleniyor: %s", recipe.id)
                    restart(recipe)
                    started.append(recipe.id)
                _AUTO_START_FAILED.pop(recipe.id, None)
                continue
            if state.get("stopped_by_user"):
                continue
            last = _AUTO_START_FAILED.get(recipe.id)
            if last and now - last < _AUTO_START_COOLDOWN:
                continue
            start(recipe)
            started.append(recipe.id)
            _AUTO_START_FAILED.pop(recipe.id, None)
        except Exception:  # noqa: BLE001 - biri patlarsa diğerleri etkilenmesin
            _AUTO_START_FAILED[recipe.id] = time.time()
            logger.warning("Eklenti otomatik başlatılamadı: %s", recipe.id, exc_info=True)
    return started


def _supervise_loop(interval: float) -> None:
    while not _SUPERVISOR_STOP.is_set():
        try:
            start_installed()
        except Exception:  # noqa: BLE001
            logger.exception("Eklenti gözcüsü hata verdi")
        if _SUPERVISOR_STOP.wait(interval):
            break


def start_supervisor(interval: float = 20.0) -> None:
    """Kurulu eklentileri hazır tutan arka plan gözcüsünü başlatır."""
    global _SUPERVISOR_THREAD
    with _LIFECYCLE_LOCK:
        if _SUPERVISOR_THREAD is not None and _SUPERVISOR_THREAD.is_alive():
            return
        _SUPERVISOR_STOP.clear()
        _SUPERVISOR_THREAD = threading.Thread(
            target=_supervise_loop,
            args=(interval,),
            name="plugin-supervisor",
            daemon=True,
        )
        _SUPERVISOR_THREAD.start()


def stop_supervisor() -> None:
    """Gözcü döngüsünü durdurur (eklenti servislerini durdurmaz)."""
    _SUPERVISOR_STOP.set()


def _running_info(recipe: PluginRecipe) -> dict:
    state = _read_state(recipe)
    return {"pid": state.get("pid", 0), "port": state.get("port", 0)}


def running_port_for_category(category: str, prefer: tuple[str, ...] = ()) -> tuple[str, int] | None:
    """Çalışan bir eklentinin (id, port) çiftini döner.

    Bir kategoride birden çok eklenti olabilir (ör. lipsync: muse_talk /
    latent_sync). `prefer` içindeki id'ler önceliklidir; yoksa kategorideki
    ilk çalışan eklenti seçilir. Çalışan yoksa None döner.
    """
    candidates = [r for r in CATALOG if r.category == category]
    ordered = [r for rid in prefer for r in candidates if r.id == rid]
    ordered += [r for r in candidates if r not in ordered]

    for recipe in ordered:
        if not is_installed(recipe) or not is_running(recipe):
            continue
        port = _read_state(recipe).get("port", 0)
        if port:
            return recipe.id, int(port)
    return None


# ------------------------------------------------------------------
# Log
# ------------------------------------------------------------------

def read_log(recipe: PluginRecipe, kind: str = "service", max_bytes: int = 200_000) -> str:
    """Servis ya da kurulum logunu döner (son max_bytes bayt)."""
    path = _install_log(recipe) if kind == "install" else _service_log(recipe)
    if not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


# ------------------------------------------------------------------
# Kaldırma
# ------------------------------------------------------------------

# Boş kalsa bile silinmemesi gereken kök klasörler
_PROTECTED_DIRS = {PROJECT_ROOT, PROJECT_ROOT / "runtime", PROJECT_ROOT / "models"}


def _prune_empty_parents(paths: list[Path]) -> None:
    """Silinen hedeflerin boş kalan ebeveyn klasörlerini temizler.

    Örn. runtime/qwen-tts/venv silindikten sonra boş kalan runtime/qwen-tts
    de kaldırılır; ancak runtime/ ve models/ korunur.
    """
    for path in paths:
        parent = path.parent
        while parent not in _PROTECTED_DIRS and parent.is_dir():
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _force_rmtree(path: Path) -> None:
    """Salt-okunur dosyalar yüzünden kalan artıkları da temizler."""
    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        return
    for entry in [path, *path.rglob("*")]:
        try:
            os.chmod(entry, 0o700)
        except OSError:
            continue
    shutil.rmtree(path, ignore_errors=True)


def remove(recipe: PluginRecipe) -> dict:
    """Eklentiyi tamamen kaldırır.

    Servisi durdurur; ardından eklentiye bağlı her şeyi siler: çalışma
    ortamı (venv), kaynak kodu, ağırlıklar ve runtime kayıtları. Kurulum
    sürerken kaldırmayı reddeder (önce İptal edilmeli). Silme sonrası
    hedefler yeniden denetlenir; kalan varsa ``failed`` ile raporlanır.
    """
    with _INSTALL_LOCK:
        status = _INSTALLS.get(recipe.id)
        if status and status.get("running"):
            raise PluginError("Kurulum sürerken kaldırılamaz; önce kurulumu iptal edin.")

    with _LIFECYCLE_LOCK:
        try:
            stop(recipe)
        except Exception:  # noqa: BLE001 - kaldırma engellenmesin
            logger.exception("Kaldırma sırasında servis durdurulamadı: %s", recipe.id)

        targets: list[Path] = _artifact_paths(recipe)

        removed: list[str] = []
        failed: list[str] = []
        for target in targets:
            if not target.exists():
                continue
            _force_rmtree(target)
            if target.exists():
                failed.append(str(target))
                logger.error("Eklenti artığı silinemedi: %s", target)
            else:
                removed.append(str(target))

        _prune_empty_parents(targets)

        _PROCS.pop(recipe.id, None)
        with _INSTALL_LOCK:
            _INSTALLS.pop(recipe.id, None)
        with _INSTALL_CANCEL_LOCK:
            _INSTALL_CANCEL.discard(recipe.id)
            _INSTALL_PROCS.pop(recipe.id, None)
        logger.info("Eklenti kaldırıldı: %s (%s)", recipe.id, ", ".join(removed) or "dosya yok")
        return {"ok": not failed, "removed": removed, "failed": failed}
