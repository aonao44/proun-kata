"""FastAPIアプリケーションのエントリポイント。"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.dependencies import get_pipeline, new_request_id
from api.schemas import (
    PHONEME_MAX_BYTES,
    FinalConsonantHandling,
    KanaStyle,
    LongVowelLevel,
    RColoring,
    ResponseParams,
    SokuonLevel,
    THMapping,
    TranscriptionResponse,
)
from asr.pipeline import PhonemePipeline, ShortAudioError
from common.config import get_settings
from kana.mapping import KanaConversionOptions, to_kana_sequence

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Phoneme to Kana API", version="0.1.0")


@app.get("/healthz", summary="ヘルスチェック")
def healthz() -> dict[str, str]:
    """バックエンドの状態を返す。"""

    return {"status": "ok"}


@app.post(
    "/transcribe_phonetic",
    response_model=TranscriptionResponse,
    summary="英語音声からカタカナ表記を生成する",
)
async def transcribe_phonetic(
    audio: UploadFile = File(..., description="16kHz mono WAV"),  # noqa: B008
    style: KanaStyle | None = Form(default=None),  # noqa: B008
    long_vowel_level: LongVowelLevel | None = Form(default=None),  # noqa: B008
    sokuon_level: SokuonLevel | None = Form(default=None),  # noqa: B008
    r_coloring: RColoring | None = Form(default=None),  # noqa: B008
    final_c_t: FinalConsonantHandling = Form(default=FinalConsonantHandling.xtsu),  # noqa: B008
    th: THMapping = Form(default=THMapping.su),  # noqa: B008
    pipeline: PhonemePipeline = Depends(get_pipeline),  # noqa: B008
    req_id: str = Depends(new_request_id),  # noqa: B008
) -> TranscriptionResponse:
    """音声を受け取り音素列→カタカナ変換した結果を返す。"""

    if audio.content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "invalid_mime"},
        )
    input_name = audio.filename or "unknown"
    payload = await audio.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "empty_audio"},
        )
    if len(payload) > PHONEME_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": "audio_too_large"},
        )

    try:
        phones, metrics = await pipeline.transcribe(payload, req_id=req_id)
    except ShortAudioError as exc:
        LOGGER.debug(
            "short_audio duration=%.2f min=%.2f", exc.duration_ms, exc.minimum_ms
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "short_audio"},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("transcription failed", extra={"req_id": req_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "asr_backend_failed"},
        ) from exc

    settings = get_settings()
    resolved_style = _resolve_style(style, settings.reading_style)
    resolved_long = _resolve_level(long_vowel_level, settings.long_vowel_level, 0, 2)
    resolved_sokuon = _resolve_level(sokuon_level, settings.sokuon_level, 0, 2)
    resolved_r = _resolve_level(r_coloring, settings.r_coloring, 0, 1)

    try:
        kana_result = to_kana_sequence(
            phones,
            options=KanaConversionOptions(
                reading_style=resolved_style.value,
                long_vowel_level=resolved_long,
                sokuon_level=resolved_sokuon,
                r_coloring=bool(resolved_r),
                th_style=th.value,
                final_c_handling=final_c_t.value,
                auto_long_vowel_ms=float(settings.auto_long_vowel_ms),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "kana conversion failed",
            extra={
                "req_id": req_id,
                "error": "kana-rule-failed",
                "exc_type": exc.__class__.__name__,
                "filename": input_name,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal",
                "detail": "kana-rule-failed",
                "filename": input_name,
            },
        )
    phoneme_payload = [
        {"p": phone.symbol, "start": phone.start, "end": phone.end, "conf": phone.confidence}
        for phone in phones
    ]
    kana_tokens = [token or "" for token in kana_result.tokens]
    kana_payload = [
        {"k": value, "start": phone.start, "end": phone.end}
        for phone, value in zip(phones, kana_tokens)
    ]
    kana_text_readable = kana_result.readable_text or kana_result.text
    kana_text_strict = kana_result.strict_text or kana_text_readable
    response = TranscriptionResponse(
        phones=phoneme_payload,
        kana=kana_payload,
        kana_text=kana_text_readable,
        kana_text_readable=kana_text_readable,
        kana_text_strict=kana_text_strict,
        kana_ops=kana_result.ops or [],
        params=ResponseParams(
            conf_threshold=metrics.conf_threshold,
            min_phone_ms=metrics.min_phone_ms,
            long_vowel_ms=float(settings.auto_long_vowel_ms),
            min_input_ms=metrics.min_input_ms,
            reject_ms=metrics.reject_ms,
        ),
    )

    LOGGER.info(
        (
            "req_id=%s model=%s@%s device=%s total_ms=%.2f inference_ms=%.2f "
            "cold_start_ms=%s chunk_ms=%s overlap=%.2f conf=%.2f "
            "min_phone_ms=%s"
        ),
        metrics.req_id,
        metrics.model_id,
        metrics.revision,
        metrics.device,
        metrics.total_ms,
        metrics.inference_ms,
        metrics.cold_start_ms,
        metrics.chunk_ms,
        metrics.overlap,
        metrics.conf_threshold,
        metrics.min_phone_ms,
    )
    return response


def _resolve_style(value: KanaStyle | None, fallback: str) -> KanaStyle:
    if value is not None:
        return value
    try:
        return KanaStyle(fallback)
    except ValueError:
        return KanaStyle.balanced


def _resolve_level(enum_value, fallback: int, minimum: int, maximum: int) -> int:
    if enum_value is not None:
        return int(enum_value)
    return max(minimum, min(int(fallback), maximum))
