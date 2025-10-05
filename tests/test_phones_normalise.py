"""Tests for phoneme normalisation utilities."""
from __future__ import annotations

from phones.normalise import normalize_symbol


def test_normalize_removes_stress_and_maps_diphthong() -> None:
    assert normalize_symbol("EY1") == "EY"
    assert normalize_symbol("tʃ") == "CH"


def test_normalize_maps_pause_tokens() -> None:
    assert normalize_symbol("<sp>") == "SP"
    assert normalize_symbol("sil") == "SIL"


def test_normalize_ignores_padding() -> None:
    assert normalize_symbol("[PAD]") == ""
