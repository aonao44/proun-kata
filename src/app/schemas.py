"""Pydantic schemas describing API inputs and outputs."""

from enum import Enum

from pydantic import BaseModel, Field


class KanaStyle(str, Enum):
    """Rendering style for kana output."""

    rough = "rough"
    loan = "loan"


class FinalConsonantHandling(str, Enum):
    """Controls how to render trailing stop consonants."""

    xtsu = "xtsu"
    tsu = "tsu"


class THMapping(str, Enum):
    """Switch between /θ/ rendering variants."""

    su = "su"
    zu = "zu"


class PhonemeSpan(BaseModel):
    """Represents a single recognised phoneme."""

    symbol: str = Field(alias="p")
    start: float
    end: float
    confidence: float | None = Field(default=None, alias="conf")

    model_config = {"populate_by_name": True, "str_strip_whitespace": True}


class KanaSpan(BaseModel):
    """Represents a rendered kana slice aligned to a phoneme."""

    kana: str = Field(alias="k")
    start: float
    end: float

    model_config = {"populate_by_name": True, "str_strip_whitespace": True}


class PhoneticTranscriptionResponse(BaseModel):
    """Full API response payload."""

    phones: list[PhonemeSpan]
    kana: list[KanaSpan]
    kana_text: str

    model_config = {"populate_by_name": True, "str_strip_whitespace": True}
