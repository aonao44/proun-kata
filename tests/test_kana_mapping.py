"""Kana conversion ruleset tests."""
from __future__ import annotations

from kana.mapping import KanaConversionOptions, to_kana_sequence


def test_balanced_make_it_sequence() -> None:
    phones = ["M", "EY", "K", "IH", "T"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text.startswith("メイ")
    assert "キ" in result.text
    assert result.text.endswith("ット")
    # Regression: avoid legacy stub string
    assert result.text != "ムェイクイッ"


def test_balanced_thanks_you_sequence() -> None:
    phones = ["TH", "AE", "NG", "K", "Y", "UW"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text.startswith("サ")
    assert "ンク" in result.text
    assert "キュ" in result.text or "キュー" in result.text


def test_pause_tokens_collapse() -> None:
    phones = ["K", "AA", "SP", "<SP>", "T", "UW"]
    result = to_kana_sequence(phones, options=KanaConversionOptions())
    assert result.text.count("・") == 1


def test_r_coloring_toggle() -> None:
    phones = ["AA", "R"]
    baseline = to_kana_sequence(phones, options=KanaConversionOptions(r_coloring=False))
    colored = to_kana_sequence(phones, options=KanaConversionOptions(r_coloring=True))
    assert baseline.text == "アー"
    assert colored.text.endswith("ル")


def test_long_vowel_level_zero_drops_length() -> None:
    phones = ["OW"]
    result = to_kana_sequence(phones, options=KanaConversionOptions(long_vowel_level=0))
    assert result.text in {"オウ", "オ"}


def test_disable_sokuon_level() -> None:
    phones = ["AE", "K"]
    result = to_kana_sequence(phones, options=KanaConversionOptions(sokuon_level=0))
    assert "ック" not in result.text
    assert result.text.endswith("アク")
