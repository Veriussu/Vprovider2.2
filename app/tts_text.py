"""TTS metin normalizasyonu.

Noktalama işaretlerini korur, gereksiz boşlukları temizler ve vurgu
işaretlerini (*...*) güvenli biçimde düz metne çevirir. Böylece hem panel
hem API üzerinden üretilen konuşmalar daha doğal ve tutarlı olur.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_tts_text(text: str) -> str:
    """TTS motorlarına verilecek metni temizler ve noktalamayı korur."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    # Vurgu işaretlerini kaldır (içerik korunur; motorlar SSML desteklemeyebilir).
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Tırnak ve çizgileri konuşma için daha doğal karakterlere çevir.
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": ", ",
        "–": ", ",
        "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Noktalama sonrası boşlukları güvenceye al (kelimeler birleşmesin).
    text = re.sub(r"([,;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"(\.)([A-ZÇĞİÖŞÜ])", r"\1 \2", text)
    # Çoklu noktalama işaretlerini tekilleştir.
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    # Tüm boşlukları tek boşluğa indir.
    text = re.sub(r"\s+", " ", text).strip()
    return text
