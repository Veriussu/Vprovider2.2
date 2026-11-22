import struct

from app.gguf_inspect import inspect_gguf, runtime_note_for


def _s(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_minimal_gguf(path, tensor_type: int):
    # magic, version, tensor_count, metadata_count
    data = [b"GGUF", struct.pack("<IQQ", 3, 1, 1)]
    # metadata: general.architecture = "qwen"
    data += [_s("general.architecture"), struct.pack("<I", 8), _s("qwen")]
    # tensor info: name, n_dims, dims, type, offset
    data += [_s("output.weight"), struct.pack("<I", 2)]
    data += [struct.pack("<QQ", 8, 8), struct.pack("<I", tensor_type), struct.pack("<Q", 0)]
    path.write_bytes(b"".join(data))


def test_inspect_gguf_detects_stock_tensor(tmp_path):
    path = tmp_path / "stock.gguf"
    _write_minimal_gguf(path, 2)

    info = inspect_gguf(path)

    assert info.tensor_types == {2}
    assert info.requires_external_runtime is False


def test_inspect_gguf_detects_prism_tensor(tmp_path):
    path = tmp_path / "bonsai.gguf"
    _write_minimal_gguf(path, 142)

    info = inspect_gguf(path)

    assert info.requires_external_runtime is True
    assert info.requires_prism_runtime is True
    assert "Prism/Bonsai" in runtime_note_for(info)
