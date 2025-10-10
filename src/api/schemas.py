"""APIの入出力スキーマ。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

PHONEME_MAX_BYTES = 20 * 1024 * 1024


class KanaStyle(str, Enum):
    raw = "raw"
    balanced = "balanced"
    natural = "natural"


class FinalConsonantHandling(str, Enum):
    xtsu = "xtsu"
    tsu = "tsu"


class THMapping(str, Enum):
    su = "su"
    zu = "zu"


class LongVowelLevel(int, Enum):
    none = 0
    final_focus = 1
    extended = 2


class SokuonLevel(int, Enum):
    none = 0
    balanced = 1
    aggressive = 2


class RColoring(int, Enum):
    off = 0
    on = 1


class PhonemeOut(BaseModel):
    symbol: str = Field(alias="p")
    start: float
    end: float
    confidence: float | None = Field(alias="conf", default=None)

    model_config = {"populate_by_name": True}


class KanaOut(BaseModel):
    kana: str = Field(alias="k")
    start: float
    end: float

    model_config = {"populate_by_name": True}


class TranscriptionResponse(BaseModel):
    phones: list[PhonemeOut]
    kana: list[KanaOut]
    kana_text: str
