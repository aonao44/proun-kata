"""Kana conversion ruleset tests."""
from __future__ import annotations

from kana.mapping import KanaConversionOptions, to_kana_sequence


def test_balanced_make_it_sequence() -> None:
    phones = ["m", "eɪ", "k", "ɪ", "t"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text.startswith("メイ")
    assert "キ" in result.text
    assert result.text.endswith("ット")
    # Regression: avoid legacy stub string
    assert result.text != "ムェイクイッ"


def test_balanced_thanks_you_sequence() -> None:
    phones = ["θ", "æ", "ŋ", "k", "j", "uː"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text.startswith("サ")
    assert any(seq in result.text for seq in ("ンク", "ング"))
    assert "キュ" in result.text or "キュー" in result.text


def test_pause_tokens_collapse() -> None:
    phones = ["k", "ɑ", "SP", "<SP>", "t", "uː"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text.count("・") == 1


def test_r_coloring_toggle() -> None:
    phones = ["ɑ", "ɹ"]
    baseline = to_kana_sequence(phones, options=KanaConversionOptions(r_coloring=False))
    colored = to_kana_sequence(phones, options=KanaConversionOptions(r_coloring=True))
    assert baseline.text == "アー"
    assert colored.text.endswith("ル")


def test_long_vowel_level_zero_drops_length() -> None:
    phones = ["oʊ"]
    result = to_kana_sequence(phones, options=KanaConversionOptions(long_vowel_level=0))
    assert result.text in {"オウ", "オ"}


def test_disable_sokuon_level() -> None:
    phones = ["æ", "k"]
    result = to_kana_sequence(phones, options=KanaConversionOptions(sokuon_level=0))
    assert "ック" not in result.text
    assert result.text.endswith("アク")


def test_leading_n_maps_to_syllable() -> None:
    phones = [("n", 0.08), ("iː", 0.24)]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.tokens[0] == "ニ"
    assert result.text.startswith("ニー")


def test_auto_long_vowel_from_duration() -> None:
    phones = [("b", 0.05), ("iː", 0.20), ("t", 0.05)]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert "イー" in result.tokens


def test_unknown_vowel_falls_back_to_a() -> None:
    phones = ["ɤ"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text == "ア"


def test_unknown_consonant_falls_back_to_sokuon() -> None:
    phones = ["ɣ"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text == "ッ"
