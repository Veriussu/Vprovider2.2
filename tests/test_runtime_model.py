# ─────────────────────────────────────────────────────────────
#  Bölüm:    Eklenti Tembel GPU Tutamacı Testleri
#  Dosya:    tests/test_runtime_model.py
#  Amaç:     LazyGpuModel'in tembel yüklediğini (ilk kullanımda),
#            kullanım boyunca tuttuğunu ve boşta bıraktığını doğrular.
# ─────────────────────────────────────────────────────────────

import importlib.util
import time
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "plugins" / "adapters" / "_runtime_model.py"
)
_spec = importlib.util.spec_from_file_location("_vp_runtime_model", _MODULE_PATH)
_runtime_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime_model)

LazyGpuModel = _runtime_model.LazyGpuModel


class _Counting:
    def __init__(self):
        self.loads = 0
        self.unloads = 0

    def loader(self):
        self.loads += 1
        return {"data": self.loads}

    def unloader(self, model):
        self.unloads += 1


def test_lazy_load_only_on_first_use():
    """Model, tutamaç kurulunca değil ilk kullanımda yüklenir."""
    counter = _Counting()
    handle = LazyGpuModel(counter.loader, counter.unloader, idle_seconds=999, name="t")

    assert handle.loaded is False
    assert counter.loads == 0

    with handle.use() as model:
        assert model == {"data": 1}
        assert handle.loaded is True
    with handle.use():
        pass
    assert counter.loads == 1  # ikinci kullanım yeniden yüklemedi


def test_idle_releases_gpu():
    """Boşta süre dolunca model bırakılır (unloader çağrılır)."""
    counter = _Counting()
    handle = LazyGpuModel(counter.loader, counter.unloader, idle_seconds=0.05, name="t")
    handle.start()

    with handle.use():
        assert handle.loaded is True

    deadline = time.time() + 3
    while handle.loaded and time.time() < deadline:
        time.sleep(0.05)
    assert handle.loaded is False
    assert counter.unloads == 1

    # Yeniden kullanımda tekrar yüklenir.
    with handle.use() as model:
        assert model == {"data": 2}
    assert counter.loads == 2


def test_busy_model_not_released():
    """Kullanım sürerken boşta-bırakma devreye girmez."""
    counter = _Counting()
    handle = LazyGpuModel(counter.loader, counter.unloader, idle_seconds=0.0, name="t")
    handle.start()

    with handle.use():
        time.sleep(0.2)
        assert handle.loaded is True
        assert counter.unloads == 0
