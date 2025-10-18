"""Normalization utilities for phoneme tokens."""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Sequence

LOGGER = logging.getLogger(__name__)

STRESS_PATTERN = re.compile(r"\d")
IGNORED_TOKENS = {"<s>", "</s>", "[pad]", "[unk]"}

SILENCE_TOKENS = {
    "<SP>": "SP",
    "SP": "SP",
    "SIL": "SIL",
    "NSN": "NSN",
    "PAU": "SIL",
}

SEQUENCE_TO_IPA = {
    "A": "ʌ",
    "AJ": "aɪ",
    "EY": "eɪ",
    "OW": "oʊ",
    "OJ": "ɔɪ",
    "OY": "ɔɪ",
    "AW": "aʊ",
    "EI": "eɪ",
    "AI": "aɪ",
    "OI": "ɔɪ",
    "Ɛ": "ɛ",
    "Ɡ": "ɡ",
    "Ɬ": "ɡ",
    "ɚ": "ɚ",
}

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AX": "ə",
    "AXR": "ɚ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "DX": "ɾ",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IX": "ɪ",
    "IY": "iː",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "Q": "ʔ",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "uː",
    "UX": "ʊ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

IPA_ALIAS_TO_IPA = {
    "i": "iː",
    "u": "uː",
    "ɝ": "ɚ",
    "g": "ɡ",
}

IPA_TO_ARPABET = {ipa: arpa for arpa, ipa in ARPABET_TO_IPA.items()}
IPA_TO_ARPABET.update(
    {
        "ɪ": "IH",
        "ɾ": "D",
        "ɑː": "AA",
        "e": "EY",
        "o": "OW",
        "ə": "AX",
        "ʌ": "AH",
        "oʊ": "OW",
        "eɪ": "EY",
        "ɔ": "AO",
    }
)

ENGLISH_CONSONANTS = {
    "p",
    "b",
    "t",
    "d",
    "k",
    "ɡ",
    "g",
    "f",
    "v",
    "θ",
    "ð",
    "s",
    "z",
    "ʃ",
    "ʒ",
    "h",
    "tʃ",
    "dʒ",
    "m",
    "n",
    "ŋ",
    "l",
    "ɹ",
    "w",
    "j",
    "ɾ",
    "ʔ",
}

ENGLISH_VOWELS = {
    "i",
    "iː",
    "ɪ",
    "e",
    "eɪ",
    "ɛ",
    "æ",
    "ɑ",
    "ɑː",
    "ɚ",
    "ə",
    "o",
    "oʊ",
    "ɔ",
    "ʊ",
    "u",
    "uː",
    "aɪ",
    "aʊ",
    "ɔɪ",
    "ʌ",
}

ENGLISH_PHONEMES = frozenset(ENGLISH_CONSONANTS | ENGLISH_VOWELS)
ALLOWED_SYMBOLS = frozenset(ENGLISH_PHONEMES | {"SIL", "SP", "NSN"})

DIRECT_SUBSTITUTIONS = {
    "ħ": ["h"],
    "kh": ["k"],
    "ɜ": ["ɚ"],
    "ɝ": ["ɚ"],
    "aː": ["ɑː"],
    "e̞": ["e"],
    "oɜ": ["oʊ"],
    "yæɜ": ["j", "æ", "ɚ"],
    "yæɚ": ["j", "æ", "ɚ"],
    "y": ["j"],
    "r": ["ɹ"],
    "ɡ": ["ɡ"],
    "g": ["ɡ"],
}

DIGRAPH_EXPANSIONS = {
    "ei": "eɪ",
    "ai": "aɪ",
    "oi": "ɔɪ",
}

NEAREST_SUBSTITUTIONS = {
    "ɱ": "m",
    "ɲ": "n",
    "ŋ̊": "ŋ",
    "ɣ": "ɡ",
    "x": "k",
    "ç": "h",
    "ʁ": "ɹ",
    "ʀ": "ɹ",
    "ʎ": "j",
    "ʝ": "j",
    "β": "v",
    "ð̞": "ð",
    "θ̠": "θ",
    "ɸ": "f",
    "ɕ": "ʃ",
    "ʑ": "ʒ",
}

UNKNOWN_PHONE_COUNTER: Counter[str] = Counter()


def normalize_symbol(token: str, *, log_unknown: bool = True) -> list[str]:
    """Normalize an arbitrary phoneme token to canonical English phonemes."""

    cleaned = token.strip()
    if not cleaned:
        return []

    lowered = cleaned.lower()
    if lowered in IGNORED_TOKENS:
        return []

    cleaned = STRESS_PATTERN.sub("", cleaned)
    upper = cleaned.upper()

    if upper in SILENCE_TOKENS:
        return [SILENCE_TOKENS[upper]]

    if upper in SEQUENCE_TO_IPA:
        canonical = SEQUENCE_TO_IPA[upper]
    elif upper in ARPABET_TO_IPA:
        canonical = ARPABET_TO_IPA[upper]
    else:
        canonical = cleaned

    canonical = IPA_ALIAS_TO_IPA.get(canonical, canonical)
    if not canonical:
        return []

    return clamp_to_english([canonical], log_unknown=log_unknown)


def clamp_to_english(phones: Sequence[str], *, log_unknown: bool = True) -> list[str]:
    """Clamp phoneme symbols to the configured English whitelist."""

    clamped: list[str] = []
    for symbol in phones:
        clamped.extend(_clamp_symbol(symbol, log_unknown=log_unknown))
    return clamped


def _clamp_symbol(symbol: str, *, log_unknown: bool) -> list[str]:
    cleaned = symbol.strip()
    if not cleaned:
        return []

    if cleaned in ALLOWED_SYMBOLS:
        return [cleaned]

    if cleaned in DIRECT_SUBSTITUTIONS:
        replacement = DIRECT_SUBSTITUTIONS[cleaned]
        if isinstance(replacement, str):
            replacement = [replacement]
        clamped: list[str] = []
        for item in replacement:
            clamped.extend(_clamp_symbol(item, log_unknown=log_unknown))
        return clamped

    if cleaned in DIGRAPH_EXPANSIONS:
        return _clamp_symbol(DIGRAPH_EXPANSIONS[cleaned], log_unknown=log_unknown)

    if " " in cleaned:
        parts = cleaned.split()
        aggregated: list[str] = []
        for part in parts:
            aggregated.extend(_clamp_symbol(part, log_unknown=log_unknown))
        if aggregated:
            return aggregated

    stripped = _strip_diacritics(cleaned)
    if stripped != cleaned:
        return _clamp_symbol(stripped, log_unknown=log_unknown)

    if cleaned in NEAREST_SUBSTITUTIONS:
        return _clamp_symbol(NEAREST_SUBSTITUTIONS[cleaned], log_unknown=log_unknown)

    fallback = nearest_english(cleaned)
    if log_unknown:
        UNKNOWN_PHONE_COUNTER[cleaned] += 1
        LOGGER.warning("unknown_phone=%s fallback=%s", cleaned, fallback)
    return [fallback]


def nearest_english(symbol: str) -> str:
    """Return a safe English fallback for an arbitrary phoneme symbol."""

    candidate = _strip_diacritics(symbol)
    if candidate in ALLOWED_SYMBOLS:
        return candidate
    if candidate in ENGLISH_PHONEMES:
        return candidate

    candidate_lower = candidate.lower()
    if candidate_lower in NEAREST_SUBSTITUTIONS:
        return NEAREST_SUBSTITUTIONS[candidate_lower]

    if _looks_like_vowel(candidate):
        return "ə"
    return "h"


def canonical_to_arpabet(symbol: str) -> str:
    """Convert a canonical IPA-like symbol back to an ARPAbet label."""

    cleaned = symbol.strip()
    if not cleaned:
        return ""

    upper = cleaned.upper()
    if upper in SILENCE_TOKENS:
        return upper

    base = IPA_ALIAS_TO_IPA.get(cleaned, cleaned)
    if base in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[base]
    lower = base.lower()
    if lower in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[lower]
    LOGGER.debug("unknown_phone=%s", symbol)
    return upper


def _strip_diacritics(symbol: str) -> str:
    normalized = unicodedata.normalize("NFD", symbol)
    filtered = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", filtered)


def _looks_like_vowel(symbol: str) -> bool:
    lowered = symbol.lower()
    vowel_markers = {
        "a",
        "e",
        "i",
        "o",
        "u",
        "ɪ",
        "ʊ",
        "æ",
        "ɑ",
        "ɒ",
        "ɔ",
        "ə",
        "ɚ",
        "ʌ",
        "ɐ",
        "ɤ",
    }
    return any(marker in lowered for marker in vowel_markers)


__all__ = [
    "normalize_symbol",
    "canonical_to_arpabet",
    "clamp_to_english",
    "nearest_english",
    "ALLOWED_SYMBOLS",
    "UNKNOWN_PHONE_COUNTER",
    "ENGLISH_VOWELS",
    "ENGLISH_CONSONANTS",
]
