"""IPA/特殊記号をARPAbetへ正規化するユーティリティ。"""
from __future__ import annotations

IPA_TO_ARPA: dict[str, str] = {
    "tʃ": "CH",
    "dʒ": "JH",
    "ʃ": "SH",
    "ʒ": "ZH",
    "θ": "TH",
    "ð": "DH",
    "ŋ": "NG",
    "ɹ": "R",
    "ɾ": "D",
    "ɪ": "IH",
    "i": "IY",
    "ʊ": "UH",
    "u": "UW",
    "eɪ": "EY",
    "aɪ": "AY",
    "oʊ": "OW",
    "ɔɪ": "OY",
    "aʊ": "AW",
    "ɜ": "ER",
    "ɝ": "ER",
    "ə": "AH",
    "ˈ": "",
    "ˌ": "",
    "æ": "AE",
    "ɑ": "AA",
    "ɔ": "AO",
    "ʌ": "AH",
    "ʔ": "Q",
    "sil": "SIL",
    "sp": "SP",
    "nsn": "NSN",
    "pau": "SIL",
    "|": "",
}

IGNORED_TOKENS = {"<s>", "</s>", "[pad]", "[unk]"}
SILENCE_TOKENS = {"SIL", "SP", "NSN"}
STRESS_DIGITS = set("0123456789")


def normalize_symbol(token: str) -> str:
    """wav2vec2が出力するトークンをARPAbet記号へ正規化する。"""

    cleaned = token.strip()
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    if lowered in IGNORED_TOKENS:
        return ""

    # ストレス番号や長音記号除去
    for digit in STRESS_DIGITS:
        cleaned = cleaned.replace(digit, "")
        lowered = lowered.replace(digit, "")
    cleaned = cleaned.replace("ː", "")
    lowered = lowered.replace("ː", "")

    if lowered in {"<sp>", "sp"}:
        return "SP"

    if lowered in IPA_TO_ARPA:
        return IPA_TO_ARPA[lowered]

    if cleaned.isalpha():
        return cleaned.upper()

    return cleaned.upper()
