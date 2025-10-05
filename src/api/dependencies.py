"""FastAPI依存性の定義。"""
from __future__ import annotations

import uuid
from functools import lru_cache

from asr.pipeline import PhonemePipeline
from asr.wav2vec2_backend import Wav2Vec2Pipeline
from common.config import get_settings


@lru_cache(maxsize=1)
def _build_pipeline() -> PhonemePipeline:
    settings = get_settings()
    if settings.backend.lower() != "w2v2":
        raise ValueError("Only the wav2vec2 backend is supported in this build")
    return Wav2Vec2Pipeline(
        model_id=settings.phoneme_model_id,
        revision=settings.phoneme_model_revision,
        device=settings.phoneme_device,
        chunk_ms=settings.chunk_ms,
        overlap=settings.overlap,
        conf_threshold=settings.confidence_threshold,
        min_phone_ms=settings.min_phone_ms,
    )


def get_pipeline() -> PhonemePipeline:
    """シングルトンの音素パイプラインを返す。"""

    return _build_pipeline()


def new_request_id() -> str:
    """リクエスト識別子を生成する。"""

    return uuid.uuid4().hex
