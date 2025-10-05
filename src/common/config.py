"""環境変数ベースの設定管理。"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション全体の設定。"""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    backend: str = Field(default="w2v2", validation_alias=AliasChoices("PHONEME_BACKEND", "BACKEND"))
    phoneme_model_id: str = Field(
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        validation_alias=AliasChoices("PHONEME_MODEL_ID", "MODEL_ID"),
    )
    phoneme_model_revision: str = Field(
        default="REPLACE_WITH_COMMIT_SHA",
        validation_alias=AliasChoices("PHONEME_MODEL_REVISION", "MODEL_REVISION"),
    )
    phoneme_device: str = Field(
        default="auto",
        validation_alias=AliasChoices("PHONEME_DEVICE", "DEVICE"),
    )
    chunk_ms: int = Field(default=320, validation_alias=AliasChoices("CHUNK_MS", "PHONEME_CHUNK_MS"))
    overlap: float = Field(
        default=0.5,
        validation_alias=AliasChoices("CHUNK_OVERLAP", "PHONEME_OVERLAP"),
    )
    confidence_threshold: float = Field(
        default=0.30,
        validation_alias=AliasChoices("CONF_THRESHOLD", "PHONEME_CONFIDENCE_THRESHOLD"),
    )
    min_phone_ms: int = Field(
        default=40,
        validation_alias=AliasChoices("MIN_PHONE_MS", "PHONEME_MIN_PHONE_MS"),
    )
    reading_style: str = Field(
        default="balanced",
        validation_alias=AliasChoices("READING_STYLE", "PHONEME_READING_STYLE"),
    )
    long_vowel_level: int = Field(
        default=1,
        validation_alias=AliasChoices("LONG_VOWEL_LEVEL", "PHONEME_LONG_VOWEL_LEVEL"),
    )
    sokuon_level: int = Field(
        default=1,
        validation_alias=AliasChoices("SOKUON_LEVEL", "PHONEME_SOKUON_LEVEL"),
    )
    r_coloring: int = Field(
        default=0,
        validation_alias=AliasChoices("R_COLORING", "PHONEME_R_COLORING"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """設定をシングルトンで取得する。"""

    return Settings()
