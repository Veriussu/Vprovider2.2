"""Accelerator (GPU) erişim eşgüdümü — VProvider geneli tek iş kuyruğu.

VProvider'daki tüm model ve eklenti işleri bu kuyruktan geçer: aynı anda
yalnızca bir iş GPU'yu kullanır. İstekler geldikleri sırada (FIFO) işlenir.

Kullanım:
    from app.runtime_coord import accelerator
    async with accelerator.acquire("llama:qwen"):
        ...  # GPU işi

Geriye dönük uyumluluk için `llm_execution_lock` adı da bu kuyruğa işaret eder
(eski LLM/TTS backend'leri `async with llm_execution_lock:` biçimini kullanır).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional


class AcceleratorQueue:
    """GPU erişimini serileştiren FIFO kuyruk (tek aktif iş)."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._owner: Optional[str] = None
        self._started_at = 0.0
        self._served = 0

    @asynccontextmanager
    async def acquire(self, owner: Optional[str] = None):
        """Kuyruktan sıra alır; iş bitince kilidi bırakır."""
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        self._owner = owner
        self._started_at = time.monotonic()
        try:
            yield self
        finally:
            self._served += 1
            self._owner = None
            self._started_at = 0.0
            self._lock.release()

    async def __aenter__(self) -> "AcceleratorQueue":
        await self._lock.acquire()
        self._owner = None
        self._started_at = time.monotonic()
        return self

    async def __aexit__(self, *exc) -> bool:
        self._served += 1
        self._owner = None
        self._started_at = 0.0
        self._lock.release()
        return False

    @property
    def busy(self) -> bool:
        """Şu anda bir iş çalışıyor mu?"""
        return self._lock.locked()

    def stats(self) -> dict:
        """Panel için kuyruk durumu."""
        return {
            "busy": self._lock.locked(),
            "waiting": self._waiting,
            "owner": self._owner,
            "served": self._served,
            "running_seconds": (
                round(time.monotonic() - self._started_at, 3)
                if self._started_at
                else 0.0
            ),
        }


accelerator = AcceleratorQueue()

# Geriye dönük uyumluluk: eski ad (LLM ve TTS backend'leri kullanır).
llm_execution_lock = accelerator
