"""Tests for phoneme normalisation utilities."""
from __future__ import annotations

from core.normalize import (
    ALLOWED_SYMBOLS,
    canonical_to_arpabet,
    clamp_to_english,
    normalize_symbol,
)


def test_normalize_removes_stress_and_keeps_ipa() -> None:
    assert normalize_symbol("EY1") == ["eɪ"]
    assert normalize_symbol("tʃ") == ["tʃ"]


def test_normalize_maps_pause_tokens() -> None:
    assert normalize_symbol("<sp>") == ["SP"]
    assert normalize_symbol("sil") == ["SIL"]


def test_normalize_ignores_padding() -> None:
    assert normalize_symbol("[PAD]") == []


def test_normalize_handles_variants_and_arpabet() -> None:
    assert normalize_symbol("A") == ["ʌ"]
    assert normalize_symbol("AJ") == ["aɪ"]
    assert normalize_symbol("Ɛ") == ["ɛ"]
    assert normalize_symbol("Ɡ") == ["ɡ"]
    assert normalize_symbol("IY") == ["iː"]
    assert normalize_symbol("CH") == ["tʃ"]


def test_normalize_applies_composite_mappings() -> None:
    assert normalize_symbol("ħ") == ["h"]
    assert normalize_symbol("kh") == ["k"]
    assert normalize_symbol("yæɜ") == ["j", "æ", "ɚ"]


def test_clamp_to_english_nearest_fallback() -> None:
    clamped = clamp_to_english(["ɱ", "ɤ"])
    assert all(symbol in ALLOWED_SYMBOLS for symbol in clamped)


def test_canonical_to_arpabet_roundtrip() -> None:
    assert canonical_to_arpabet("iː") == "IY"
    assert canonical_to_arpabet("u") == "UW"
    assert canonical_to_arpabet("ɾ") == "D"
