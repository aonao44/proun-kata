"""FastAPI依存性の定義。"""
from __future__ import annotations

import uuid
import logging

from asr.pipeline import PhonemePipeline
from asr.wav2vec2_backend import Wav2Vec2Pipeline
from common.config import get_settings

LOGGER = logging.getLogger(__name__)


_PIPELINE: PhonemePipeline | None = None


def _new_pipeline() -> PhonemePipeline:
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

    global _PIPELINE
    pipeline = _PIPELINE
    if pipeline is None or not hasattr(pipeline, "ensure_ready"):
        if pipeline is not None:
            LOGGER.warning("get_pipeline: legacy pipeline detected; rebuilding instance")
        pipeline = _PIPELINE = _new_pipeline()

    return pipeline


def new_request_id() -> str:
    """リクエスト識別子を生成する。"""

    return uuid.uuid4().hex
