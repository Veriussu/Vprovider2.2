# ------------------------------------------------------------------
# External OpenAI-compatible llama.cpp runtimes
# ------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

from app.config import settings
from app.gguf_inspect import GgufInfo, inspect_gguf, runtime_note_for
from app.model_manager import ModelInfo
from app.runtime_coord import llm_execution_lock

logger = logging.getLogger("vprovider")


class ExternalRuntimeUnavailable(RuntimeError):
    """Raised when a model requires a runtime that is not configured."""


def _candidate_prism_bins() -> list[Path]:
    configured = [Path(settings.prism_llama_server_bin)] if settings.prism_llama_server_bin else []
    defaults = [
        Path("runtime/prism/bin/llama-server"),
        Path("runtime/prism/llama-server"),
        Path("runtime/bonsai/bin/llama-server"),
        Path("runtime/bonsai/llama-server"),
    ]
    return configured + defaults


def prism_server_bin() -> Path | None:
    for path in _candidate_prism_bins():
        if path and path.is_file():
            return path
    return None


def _free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((settings.prism_llama_host, port))
            except OSError:
                continue
            return port
    raise ExternalRuntimeUnavailable("Harici runtime için boş port bulunamadı.")


@dataclass
class ExternalModelRuntime:
    model_id: str
    path: Path
    gguf: GgufInfo
    bin_path: Path
    port: int
    process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{settings.prism_llama_host}:{self.port}/v1"

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def ensure_started(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        cmd = [
            str(self.bin_path),
            "-m",
            str(self.path),
            "--host",
            settings.prism_llama_host,
            "--port",
            str(self.port),
            "-c",
            str(settings.context_size),
            "--alias",
            self.model_id,
        ]
        extra = shlex.split(settings.prism_llama_extra_args or "")
        cmd.extend(extra)

        log_dir = Path("runtime")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (log_dir / f"external-{self.model_id}.log").open("ab")
        logger.info("Harici llama-server başlatılıyor: %s", " ".join(cmd))
        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + 90
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ExternalRuntimeUnavailable(
                        f"Harici runtime hemen kapandı. Log: runtime/external-{self.model_id}.log"
                    )
                try:
                    resp = await client.get(f"{self.base_url}/models")
                    if resp.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise ExternalRuntimeUnavailable(
            f"Harici runtime zamanında hazır olmadı. Log: runtime/external-{self.model_id}.log"
        )

    async def post_json(self, endpoint: str, payload: dict) -> dict:
        async with llm_execution_lock:
            try:
                await self.ensure_started()
                async with httpx.AsyncClient(timeout=None) as client:
                    resp = await client.post(f"{self.base_url}{endpoint}", json=payload)
                    resp.raise_for_status()
                    return resp.json()
            finally:
                await self.stop()

    async def stream(self, endpoint: str, payload: dict) -> AsyncIterator[str]:
        async with llm_execution_lock:
            try:
                await self.ensure_started()
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST", f"{self.base_url}{endpoint}", json=payload
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
                            else:
                                yield "\n"
            finally:
                await self.stop()

    async def stop(self) -> bool:
        if not self.is_running:
            self.process = None
            return False
        process = self.process
        process.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await asyncio.to_thread(process.wait)
        self.process = None
        return True


_runtimes: dict[str, ExternalModelRuntime] = {}


def classify_external(info: ModelInfo) -> GgufInfo:
    return inspect_gguf(info.path)


def needs_external_runtime(info: ModelInfo) -> bool:
    try:
        return classify_external(info).requires_external_runtime
    except Exception:
        return False


def runtime_for(info: ModelInfo) -> ExternalModelRuntime:
    gguf = classify_external(info)
    if not gguf.requires_external_runtime:
        raise ExternalRuntimeUnavailable("Bu model harici runtime gerektirmiyor.")

    bin_path = prism_server_bin() if gguf.requires_prism_runtime else None
    if bin_path is None:
        note = runtime_note_for(gguf)
        searched = ", ".join(str(p) for p in _candidate_prism_bins())
        raise ExternalRuntimeUnavailable(
            f"{info.model_id} stock llama.cpp ile açılamaz. {note} "
            f"Prism/Bonsai uyumlu llama-server binary'sini PRISM_LLAMA_SERVER_BIN ile verin "
            f"veya şu yollardan birine koyun: {searched}"
        )

    runtime = _runtimes.get(info.model_id)
    if runtime is None:
        runtime = ExternalModelRuntime(
            model_id=info.model_id,
            path=info.path,
            gguf=gguf,
            bin_path=bin_path,
            port=_free_port(settings.prism_llama_port_start),
        )
        _runtimes[info.model_id] = runtime
    return runtime


def external_runtime_loaded(model_id: str) -> bool:
    runtime = _runtimes.get(model_id)
    return bool(runtime and runtime.is_running)


async def unload_external_runtime(model_id: str) -> bool:
    runtime = _runtimes.pop(model_id, None)
    if runtime is None:
        return False
    return await runtime.stop()


async def shutdown_external_runtimes() -> None:
    runtimes = list(_runtimes.values())
    _runtimes.clear()
    for runtime in runtimes:
        await runtime.stop()
