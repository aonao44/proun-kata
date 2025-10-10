"""音声→音素変換パイプラインのインタフェース。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class PhonemeSpan:
    """音素とタイムスタンプを表す。"""

    symbol: str
    start: float
    end: float
    confidence: float | None


@dataclass(slots=True)
class TranscriptionMetrics:
    """推論計測値をまとめる。"""

    req_id: str
    model_id: str
    revision: str
    device: str
    cold_start_ms: float | None
    inference_ms: float
    total_ms: float
    chunk_ms: int
    overlap: float
    conf_threshold: float
    min_phone_ms: float


class PhonemePipeline(Protocol):
    """wav2vec2ベースのパイプラインが満たすべきインタフェース。"""

    async def transcribe(self, audio_bytes: bytes, *, req_id: str) -> tuple[list[PhonemeSpan], TranscriptionMetrics]:
        """音声バイト列を音素系列に変換する。"""

        raise NotImplementedError
