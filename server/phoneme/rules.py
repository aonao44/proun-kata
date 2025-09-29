"""Rule-based mapping from phoneme symbols to kana tokens."""
from __future__ import annotations

from typing import Iterable

PROMPT_VOWELS = {"EY", "AY", "OW", "OY", "AW"}
SILENCE_LABELS = {"SIL", "SP", "NSN"}


def to_kana_sequence(
    phones: Iterable[str],
    *,
    kana_style: str,
    final_c_t: str,
    th_style: str,
) -> list[str]:
    """Convert a sequence of phoneme labels into kana tokens."""

    phone_list = list(phones)
    last_content_index = _last_content_index(phone_list)

    tokens: list[str] = []
    for idx, phone in enumerate(phone_list):
        if phone in SILENCE_LABELS:
            tokens.append("")
            continue
        kana = _map_phone(
            phone,
            is_last_content=idx == last_content_index,
            kana_style=kana_style,
            final_c_t=final_c_t,
            th_style=th_style,
        )
        tokens.append(kana)
    return tokens


def _map_phone(
    phone: str,
    *,
    is_last_content: bool,
    kana_style: str,
    final_c_t: str,
    th_style: str,
) -> str:
    phone_upper = phone.upper()

    if phone_upper == "TH":
        return "ス" if th_style == "su" else "ズ"

    if phone_upper == "ER":
        return "ァール"

    if phone_upper in PROMPT_VOWELS:
        return _map_diphthong(phone_upper, kana_style)

    if phone_upper == "T" and is_last_content:
        return "ッ" if final_c_t == "xtsu" else "ツ"

    base_mapping = {
        "M": "ム",
        "EY": "ェイ",
        "K": "ク",
        "IH": "イ",
        "P": "プ",
        "S": "ス",
        "N": "ン",
        "R": "ル",
        "L": "ル",
        "D": "ド",
        "AY": "アイ",
        "OW": "オウ",
        "OY": "オイ",
        "AW": "アウ",
        "B": "ブ",
        "G": "グ",
        "F": "フ",
        "V": "ヴ",
        "Z": "ズ",
        "SH": "シュ",
        "CH": "チ",
        "JH": "ジ",
        "Y": "イ",
        "W": "ウ",
        "HH": "ハ",
        "EH": "エ",
        "AH": "ア",
        "UH": "ウ",
        "UW": "ウー",
        "IY": "イー",
        "AA": "アー",
        "AE": "ア",
        "AO": "オー",
        "NG": "ング",
    }

    kana = base_mapping.get(phone_upper, phone_upper)

    if phone_upper == "EY" and kana_style == "loan":
        kana = "メイ"
    elif phone_upper == "AY" and kana_style == "loan":
        kana = "アイ"

    return kana


def _map_diphthong(phone: str, kana_style: str) -> str:
    if phone == "EY":
        return "メイ" if kana_style == "loan" else "ェイ"
    if phone == "AY":
        return "アイ"
    if phone == "OW":
        return "オウ"
    if phone == "OY":
        return "オイ"
    if phone == "AW":
        return "アウ"
    return phone


def _last_content_index(phones: list[str]) -> int:
    for idx in range(len(phones) - 1, -1, -1):
        if phones[idx] not in SILENCE_LABELS:
            return idx
    return -1
