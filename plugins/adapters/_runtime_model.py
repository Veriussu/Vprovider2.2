#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  Bölüm:    Eklenti Servisleri için Ortak GPU Model Tutamacı
#  Dosya:    plugins/adapters/_runtime_model.py
#  Amaç:     Eklenti servis süreçleri ayakta kalır ama modeli tembel
#            yükler: ilk istek gelince GPU'ya alır, bir süre
#            kullanılmazsa GPU'yu bırakır (VRAM'i serbest bırakır).
#            Böylece tek GPU'da eklentiler sırayla çalışabilir.
#  Kullanım: handle = LazyGpuModel(loader, unloader, idle_seconds=120)
#            with handle.use() as model: ...
# ─────────────────────────────────────────────────────────────

import gc
import threading
import time
from contextlib import contextmanager


def free_torch_model(model) -> None:
    """Bir torch modelini CPU'ya taşıyıp GPU belleğini boşaltır."""
    try:
        move = getattr(model, "to", None)
        if callable(move):
            move("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class LazyGpuModel:
    """Tembel yüklenen, boşta GPU'yu bırakan model tutamacı."""

    def __init__(self, loader, unloader, idle_seconds: float = 120.0, name: str = "model"):
        self._loader = loader          # () -> model
        self._unloader = unloader      # (model) -> None
        # Negatif değer boşta-bırakma gözlemcisini kapatır (sürekli aktif).
        self._idle = float(idle_seconds)
        self._name = name
        self._model = None
        self._last_use = 0.0
        self._busy = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def start(self) -> None:
        """Boşta-bırakma gözlemcisini başlatır."""
        thread = threading.Thread(target=self._reaper, name="gpu-reaper", daemon=True)
        thread.start()

    @contextmanager
    def use(self):
        """Modeli (gerekiyorsa) yükler ve kullanım boyunca tutar."""
        with self._lock:
            if self._model is None:
                print(f"[{self._name}] model GPU'ya yükleniyor...", flush=True)
                self._model = self._loader()
                print(f"[{self._name}] model hazır.", flush=True)
            self._busy += 1
            self._last_use = time.time()
        try:
            yield self._model
        finally:
            with self._lock:
                self._busy = max(0, self._busy - 1)
                self._last_use = time.time()
                # idle <= 0 (istek ile aktif / dynamic) ise kullanım biter bitmez
                # modeli GPU'dan bırak; böylece VRAM anında boşalır.
                if self._idle <= 0 and self._busy == 0 and self._model is not None:
                    self._free_locked()

    def free(self) -> None:
        """Modeli hemen GPU'dan bırakır."""
        with self._lock:
            self._free_locked()

    def _free_locked(self) -> None:
        model = self._model
        self._model = None
        self._last_use = 0.0
        if model is not None and self._unloader is not None:
            print(f"[{self._name}] boşta; GPU bırakılıyor.", flush=True)
            try:
                self._unloader(model)
            except Exception as exc:  # noqa: BLE001
                print(f"[{self._name}] boşaltma hatası: {exc}", flush=True)

    def _reaper(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                if (
                    self._model is not None
                    and self._busy == 0
                    and self._idle >= 0
                    and self._last_use
                    and (time.time() - self._last_use) >= self._idle
                ):
                    self._free_locked()
