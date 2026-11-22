# ─────────────────────────────────────────────────────────────
#  Bölüm:    Birleşik Model Sayaç (Registry) Testleri
#  Dosya:    tests/test_registry.py
#  Amaç:     app/registry.py içindeki tip (kind) ve kategori (category)
#            sınıflandırmasını geçici disk yapısı üzerinde doğrular.
#  Mekanik:  - Geçici models/ klasörüne örnek model dizinleri kurulur.
#            - scan_typed() her biçimi ve kategoriyi doğru bulur.
#            - category_for_gguf() klasör/ad işaretlerini değerlendirir.
#            - delete_typed_model() yalnızca ilgili dizini siler.
# ─────────────────────────────────────────────────────────────

import json

import pytest

from app import registry


def _write_pipeline(base, name, index):
    """diffusers pipeline dizini (model_index.json) oluşturur.

    index: model_index.json içeriği olduğu gibi yazılır (yapıyı sınama
    bırakılır: hem eski düz hem yeni iç içe "class_name" biçimi test edilir).
    """
    d = base / registry.KIND_SAFETENSORS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_index.json").write_text(json.dumps(index), "utf-8")
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    return d


def _write_st(base, name):
    """sentence-transformers model dizini oluşturur."""
    d = base / registry.KIND_EMBEDDINGS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "bert"}), "utf-8")
    (d / "model.safetensors").write_bytes(b"x")
    return d


def _write_ct2(base, name):
    """ctranslate2 (faster-whisper) dizini oluşturur."""
    d = base / registry.KIND_CT2 / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model": "whisper"}), "utf-8")
    (d / "model.bin").write_bytes(b"x")
    return d


def _write_onnx(base, name):
    """ONNX aktarım dizini oluşturur."""
    d = base / registry.KIND_ONNX / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(b"x")
    return d


# ------------------------------------------------------------------
# Tip bazlı tarama
# ------------------------------------------------------------------

def test_scan_typed_detects_all_kinds(tmp_path):
    """Her biçim (safetensors/onnx/ct2/embeddings) doğru kategoriyle bulunur."""
    models = tmp_path / "models"
    _write_pipeline(models, "sd-turbo", {"_class_name": "StableDiffusionPipeline"})
    _write_pipeline(
        models, "animate",
        {"text_encoder": {"class_name": "CLIPTextModel"},
         "unet": {"class_name": "UNet3DConditionModel"}},
    )
    _write_pipeline(models, "musicgen", {"_class_name": "MusicgenPipeline"})
    _write_onnx(models, "whisper-onnx")
    _write_ct2(models, "whisper-small")
    _write_st(models, "all-MiniLM-L6-v2")

    found = {e.model_id: e for e in registry.scan_typed(models_dir=models)}

    assert found["sd-turbo"].kind == registry.KIND_SAFETENSORS
    assert found["sd-turbo"].category == registry.CAT_IMAGE
    assert found["animate"].category == registry.CAT_VIDEO
    assert found["musicgen"].category == registry.CAT_MUSIC
    assert found["whisper-onnx"].kind == registry.KIND_ONNX
    assert found["whisper-onnx"].category == registry.CAT_GENERIC
    assert found["whisper-small"].kind == registry.KIND_CT2
    assert found["whisper-small"].category == registry.CAT_STT
    assert found["all-MiniLM-L6-v2"].kind == registry.KIND_EMBEDDINGS
    assert found["all-MiniLM-L6-v2"].category == registry.CAT_EMBEDDING
    assert found["all-MiniLM-L6-v2"].size_bytes > 0


def test_scan_typed_ignores_partial_dirs(tmp_path):
    """Sözleşmeye uymayan dizinler (ör. yalnızca config.json) listelenmez."""
    models = tmp_path / "models"
    d = models / "embeddings" / "yarim"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", "utf-8")   # ağırlık dosyası yok
    _write_st(models, "tam")

    ids = {e.model_id for e in registry.scan_typed(models_dir=models)}
    assert ids == {"tam"}


def test_scan_typed_missing_dir_is_empty(tmp_path):
    assert registry.scan_typed(models_dir=tmp_path / "yok") == []


def test_scan_typed_detects_category_dirs(tmp_path):
    """Kategori klasörleri (image/video/music/tts/...) biçimle bulunur."""
    models = tmp_path / "models"

    # image: diffusers pipeline doğrudan models/image/ altında
    img = models / "image" / "sd-turbo"
    img.mkdir(parents=True)
    (img / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), "utf-8"
    )
    (img / "diffusion_pytorch_model.safetensors").write_bytes(b"x")

    # tts: transformers ağırlığı (config.json + safetensors)
    tts = models / "tts" / "kokoro"
    tts.mkdir(parents=True)
    (tts / "config.json").write_text(json.dumps({"model_type": "kokoro"}), "utf-8")
    (tts / "model.safetensors").write_bytes(b"x")

    # stt: ctranslate2 dizini
    stt = models / "stt" / "faster-whisper"
    stt.mkdir(parents=True)
    (stt / "config.json").write_text(json.dumps({"model": "whisper"}), "utf-8")
    (stt / "model.bin").write_bytes(b"x")

    # music: diffusers pipeline
    mus = models / "music" / "musicgen"
    mus.mkdir(parents=True)
    (mus / "model_index.json").write_text(
        json.dumps({"_class_name": "MusicgenPipeline"}), "utf-8"
    )
    (mus / "diffusion_pytorch_model.safetensors").write_bytes(b"x")

    found = {e.model_id: e for e in registry.scan_typed(models_dir=models)}

    assert found["sd-turbo"].category == registry.CAT_IMAGE
    assert found["sd-turbo"].kind == registry.KIND_SAFETENSORS
    assert found["kokoro"].category == "tts"
    assert found["faster-whisper"].category == registry.CAT_STT
    assert found["faster-whisper"].kind == registry.KIND_CT2
    assert found["musicgen"].category == registry.CAT_MUSIC


# ------------------------------------------------------------------
# GGUF kategorisi (reasoning)
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel_path,expected",
    [
        ("model.gguf", registry.CAT_CHAT),
        ("reasoning/deepseek-r1-Q4_K_M.gguf", registry.CAT_REASONING),
        ("qwq-32b-thinking.gguf", registry.CAT_REASONING),
        ("my-R1-Model.gguf", registry.CAT_REASONING),
        ("llama-3.1-8b.gguf", registry.CAT_CHAT),
    ],
)
def test_category_for_gguf(tmp_path, rel_path, expected):
    """Klasör sözleşmesi ve ad deseni reasoning kategorisini belirler."""
    models = tmp_path / "models"
    path = models / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    assert registry.category_for_gguf(path.stem, path, models) == expected


def test_get_typed_entry_by_name(tmp_path):
    """Tip bazlı model adıyla bulunur; olmayan ad None döner."""
    models = tmp_path / "models"
    _write_st(models, "embed-model")
    entry = registry.get_typed_entry("embed-model", models_dir=models)
    assert entry is not None and entry.kind == registry.KIND_EMBEDDINGS
    assert registry.get_typed_entry("yok", models_dir=models) is None


# ------------------------------------------------------------------
# Silme
# ------------------------------------------------------------------

def test_delete_typed_model_removes_only_that_dir(tmp_path):
    """Silme yalnızca hedef model dizinini kaldırır, diğerleri korunur."""
    models = tmp_path / "models"
    _write_st(models, "silinecek")
    keep = _write_st(models, "korunacak")

    removed = registry.delete_typed_model("silinecek", models_dir=models)

    assert removed and "silinecek" in removed[0]
    assert not (models / "embeddings" / "silinecek").exists()
    assert keep.exists()


def test_delete_typed_model_missing_is_empty(tmp_path):
    assert registry.delete_typed_model("yok", models_dir=tmp_path) == []


def test_delete_typed_model_removes_category_dir(tmp_path):
    """Silme kategori klasörlerindeki (image/music/...) dizinleri de kaldırır."""
    models = tmp_path / "models"
    d = models / "music" / "musicgen"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text(json.dumps({"_class_name": "MusicgenPipeline"}), "utf-8")

    removed = registry.delete_typed_model("musicgen", models_dir=models)

    assert removed and "musicgen" in removed[0]
    assert not d.exists()

