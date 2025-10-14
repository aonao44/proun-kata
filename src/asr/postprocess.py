"""Post-processing heuristics to refine decoded phoneme spans."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from core.normalize import ENGLISH_VOWELS

from .pipeline import PhonemeSpan

SILENCE_SYMBOLS = {"SIL", "SP", "NSN"}
VOICING_MAP = {
    "p": "b",
    "t": "d",
    "k": "ɡ",
    "f": "v",
    "s": "z",
    "θ": "ð",
}
VOICED_NEIGHBORS = (
    ENGLISH_VOWELS
    | {
        "m",
        "n",
        "ŋ",
        "l",
        "ɹ",
        "w",
        "j",
        "ɾ",
        "ð",
        "v",
        "z",
        "ʒ",
        "ɚ",
        "ə",
    }
)
FLAP_TARGETS = {"t", "d"}
TH_SYMBOLS = {"θ", "ð"}


def apply_phonetic_postprocessing(spans: Sequence[PhonemeSpan]) -> list[PhonemeSpan]:
    """Apply heuristic adjustments to better match American English pronunciation."""

    if not spans:
        return []

    adjusted = list(spans)
    adjusted = _apply_flap_rules(adjusted)
    adjusted = _apply_voicing_adjustments(adjusted)
    adjusted = _recover_missing_vowels(adjusted)
    return adjusted


def _apply_flap_rules(spans: Sequence[PhonemeSpan]) -> list[PhonemeSpan]:
    result = list(spans)
    for idx in range(1, len(result) - 1):
        current = result[idx]
        if current.symbol not in FLAP_TARGETS:
            continue
        prev_symbol = result[idx - 1].symbol
        next_symbol = result[idx + 1].symbol
        if _is_vowel(prev_symbol) and (_is_vowel(next_symbol) or next_symbol == "ɚ"):
            result[idx] = replace(current, symbol="ɾ")
    return result


def _apply_voicing_adjustments(spans: Sequence[PhonemeSpan]) -> list[PhonemeSpan]:
    result = list(spans)
    for idx, span in enumerate(result):
        symbol = span.symbol
        replacement = None
        if symbol in VOICING_MAP:
            confidence = span.confidence if span.confidence is not None else 0.0
            if confidence < 0.45 and _has_voiced_environment(result, idx):
                replacement = VOICING_MAP[symbol]
        elif symbol in TH_SYMBOLS:
            continue
        else:
            if symbol in {"d", "f"} and _near_th_environment(result, idx):
                replacement = "ð" if symbol == "d" else "θ"
        if replacement and replacement != symbol:
            result[idx] = replace(span, symbol=replacement)
    return result


def _recover_missing_vowels(spans: Sequence[PhonemeSpan]) -> list[PhonemeSpan]:
    result = list(spans)
    idx = 0
    while idx < len(result):
        if not _is_content_symbol(result[idx].symbol) or _is_vowel(result[idx].symbol):
            idx += 1
            continue
        start = idx
        while idx < len(result) and _is_content_symbol(result[idx].symbol) and not _is_vowel(result[idx].symbol):
            idx += 1
        cluster_len = idx - start
        if cluster_len >= 3:
            target_idx = min(
                range(start, idx),
                key=lambda i: result[i].confidence if result[i].confidence is not None else 0.0,
            )
            target = result[target_idx]
            result[target_idx] = replace(target, symbol="ə")
    return result


def _is_vowel(symbol: str) -> bool:
    return symbol in ENGLISH_VOWELS


def _is_content_symbol(symbol: str) -> bool:
    return symbol not in SILENCE_SYMBOLS


def _has_voiced_environment(spans: Sequence[PhonemeSpan], index: int) -> bool:
    prev_symbol = spans[index - 1].symbol if index > 0 else None
    next_symbol = spans[index + 1].symbol if index + 1 < len(spans) else None
    neighbors = [prev_symbol, next_symbol]
    return any(sym in VOICED_NEIGHBORS if sym else False for sym in neighbors)


def _near_th_environment(spans: Sequence[PhonemeSpan], index: int) -> bool:
    prev_symbol = spans[index - 1].symbol if index > 0 else None
    next_symbol = spans[index + 1].symbol if index + 1 < len(spans) else None
    return any(sym in TH_SYMBOLS for sym in (prev_symbol, next_symbol))


__all__ = ["apply_phonetic_postprocessing"]
