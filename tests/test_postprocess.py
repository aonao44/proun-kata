"""Unit tests for phonetic post-processing heuristics."""
from __future__ import annotations

from asr.pipeline import PhonemeSpan
from asr.postprocess import apply_phonetic_postprocessing


def _span(symbol: str, start: float = 0.0, end: float = 0.03, confidence: float | None = 0.5) -> PhonemeSpan:
    return PhonemeSpan(symbol=symbol, start=start, end=end, confidence=confidence)


def test_flap_t_between_vowels() -> None:
    spans = [_span("ɑ"), _span("t", confidence=0.6), _span("ə")]
    result = apply_phonetic_postprocessing(spans)
    assert result[1].symbol == "ɾ"


def test_voicing_adjustment_in_voiced_environment() -> None:
    spans = [_span("ɑ"), _span("k", confidence=0.2), _span("ə")]
    result = apply_phonetic_postprocessing(spans)
    assert result[1].symbol == "ɡ"


def test_recover_missing_vowel_in_consonant_cluster() -> None:
    spans = [
        _span("s", confidence=0.8),
        _span("t", confidence=0.1),
        _span("r", confidence=0.3),
        _span("s", confidence=0.2),
    ]
    result = apply_phonetic_postprocessing(spans)
    assert any(span.symbol == "ə" for span in result)


def test_boundary_glottal_removed_when_low_conf() -> None:
    spans = [_span("ʔ", confidence=0.1), _span("æ", confidence=0.6), _span("ʔ", confidence=0.2)]
    result = apply_phonetic_postprocessing(spans)
    assert result[0].symbol != "ʔ"
    assert result[-1].symbol != "ʔ"
