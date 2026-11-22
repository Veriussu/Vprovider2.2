# ─────────────────────────────────────────────────────────────
#  Bölüm:    GPU Erişim Kuyruğu Testleri
#  Dosya:    tests/test_runtime_coord.py
#  Amaç:     VProvider geneli tek iş kuyruğunun (accelerator) işleri
#            serileştirdiğini (aynı anda tek iş) ve FIFO sırasını
#            koruduğunu doğrular.
# ─────────────────────────────────────────────────────────────

import asyncio

from app.runtime_coord import accelerator, llm_execution_lock


def test_lock_alias_points_to_accelerator():
    """Geriye dönük `llm_execution_lock` adı kuyruğa işaret eder."""
    assert llm_execution_lock is accelerator


def test_jobs_run_one_at_a_time_fifo():
    """İşler aynı anda değil, kuyruğa giriş sırasına göre çalışır."""
    events: list[tuple[str, str]] = []

    async def job(name: str, delay: float):
        async with accelerator.acquire(name):
            events.append(("start", name))
            await asyncio.sleep(delay)
            events.append(("end", name))

    async def scenario():
        await asyncio.gather(
            job("a", 0.05),
            job("b", 0.01),
            job("c", 0.01),
        )

    asyncio.run(scenario())
    # Hiçbir iş diğerinin ortasına girmez: start/end çiftleri bitişik.
    assert events == [
        ("start", "a"), ("end", "a"),
        ("start", "b"), ("end", "b"),
        ("start", "c"), ("end", "c"),
    ]


def test_stats_report_owner_and_served():
    """Kuyruk istatistikleri sahibi ve hizmet edilen iş sayısını bildirir."""
    seen = {}

    async def job():
        async with accelerator.acquire("iş-x"):
            seen["busy"] = accelerator.busy
            seen["owner"] = accelerator.stats()["owner"]

    asyncio.run(job())
    assert seen["busy"] is True
    assert seen["owner"] == "iş-x"
    assert accelerator.busy is False
    assert accelerator.stats()["owner"] is None
