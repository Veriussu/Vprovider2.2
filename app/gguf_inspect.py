# ------------------------------------------------------------------
# GGUF metadata inspection
# ------------------------------------------------------------------

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO


GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

STOCK_GGML_TYPE_MAX_EXCLUSIVE = 43
PRISM_TENSOR_TYPES = {142: "PQ2_0", 143: "PTQ1_0"}


@dataclass
class GgufInfo:
    """Small GGUF header summary used to pick a compatible runtime."""

    path: Path
    version: int = 0
    tensor_count: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    tensor_types: set[int] = field(default_factory=set)

    @property
    def unknown_tensor_types(self) -> set[int]:
        return {t for t in self.tensor_types if t >= STOCK_GGML_TYPE_MAX_EXCLUSIVE}

    @property
    def prism_tensor_types(self) -> set[int]:
        return self.tensor_types & set(PRISM_TENSOR_TYPES)

    @property
    def requires_prism_runtime(self) -> bool:
        return bool(self.prism_tensor_types)

    @property
    def requires_external_runtime(self) -> bool:
        return bool(self.unknown_tensor_types)


class GgufReadError(ValueError):
    """Raised when a file is not a readable GGUF header."""


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GgufReadError("GGUF başlığı beklenenden kısa.")
    return data


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _gguf_string(f: BinaryIO) -> str:
    n = _u64(f)
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip(f: BinaryIO, n: int) -> None:
    if n < 0:
        raise GgufReadError("Geçersiz GGUF uzunluğu.")
    if f.seekable():
        f.seek(n, 1)
    else:
        _read_exact(f, n)


def _read_value(f: BinaryIO, value_type: int):
    if value_type in (GGUF_VALUE_UINT8, GGUF_VALUE_INT8, GGUF_VALUE_BOOL):
        return _read_exact(f, 1)[0]
    if value_type in (GGUF_VALUE_UINT16, GGUF_VALUE_INT16):
        _skip(f, 2)
        return None
    if value_type in (GGUF_VALUE_UINT32, GGUF_VALUE_INT32, GGUF_VALUE_FLOAT32):
        raw = _read_exact(f, 4)
        if value_type == GGUF_VALUE_UINT32:
            return struct.unpack("<I", raw)[0]
        if value_type == GGUF_VALUE_INT32:
            return struct.unpack("<i", raw)[0]
        return struct.unpack("<f", raw)[0]
    if value_type in (GGUF_VALUE_UINT64, GGUF_VALUE_INT64, GGUF_VALUE_FLOAT64):
        raw = _read_exact(f, 8)
        if value_type == GGUF_VALUE_UINT64:
            return struct.unpack("<Q", raw)[0]
        if value_type == GGUF_VALUE_INT64:
            return struct.unpack("<q", raw)[0]
        return struct.unpack("<d", raw)[0]
    if value_type == GGUF_VALUE_STRING:
        return _gguf_string(f)
    if value_type == GGUF_VALUE_ARRAY:
        item_type = _u32(f)
        count = _u64(f)
        values = []
        keep_values = count <= 64 and item_type in (
            GGUF_VALUE_UINT32,
            GGUF_VALUE_INT32,
            GGUF_VALUE_UINT64,
            GGUF_VALUE_INT64,
            GGUF_VALUE_FLOAT32,
            GGUF_VALUE_FLOAT64,
            GGUF_VALUE_STRING,
        )
        for _ in range(count):
            value = _read_value(f, item_type)
            if keep_values:
                values.append(value)
        return values if keep_values else None
    raise GgufReadError(f"Bilinmeyen GGUF metadata tipi: {value_type}")


def inspect_gguf(path: str | Path) -> GgufInfo:
    """Read enough of a GGUF file to know its tensor type IDs.

    The function intentionally does not load tensors, so it is cheap compared
    with model initialization and can run on every model listing/request.
    """

    path = Path(path)
    with path.open("rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise GgufReadError(f"{path} GGUF dosyası değil.")
        version = _u32(f)
        tensor_count = _u64(f)
        metadata_count = _u64(f)
        info = GgufInfo(path=path, version=version, tensor_count=tensor_count)

        for _ in range(metadata_count):
            key = _gguf_string(f)
            value_type = _u32(f)
            try:
                info.metadata[key] = _read_value(f, value_type)
            except GgufReadError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GgufReadError(f"GGUF metadata okunamadı: {key}") from exc

        for _ in range(tensor_count):
            _name = _gguf_string(f)
            dims = _u32(f)
            for _ in range(dims):
                _u64(f)
            info.tensor_types.add(_u32(f))
            _u64(f)  # tensor data offset
        return info


def runtime_note_for(info: GgufInfo) -> str:
    """Human-readable runtime hint for model list and errors."""

    if info.prism_tensor_types:
        names = ", ".join(PRISM_TENSOR_TYPES[t] for t in sorted(info.prism_tensor_types))
        return f"Prism/Bonsai llama.cpp runtime gerekli ({names})."
    if info.unknown_tensor_types:
        ids = ", ".join(str(t) for t in sorted(info.unknown_tensor_types))
        return f"Stock llama.cpp bu GGUF tensor tiplerini desteklemiyor: {ids}."
    return "Stock llama.cpp uyumlu."
