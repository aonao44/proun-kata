"""FastAPI entrypoint for the phonetic transcription service."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from phoneme.pipeline import KanaToken, PhonemePipeline, PhonemeResult
from phoneme.rules import to_kana_sequence

from .dependencies import get_pipeline
from .schemas import (
    FinalConsonantHandling,
    KanaSpan,
    KanaStyle,
    PhonemeSpan,
    PhoneticTranscriptionResponse,
    THMapping,
)

app = FastAPI(
    title="Phonetic Transcription Service",
    description="Converts raw speech audio to katakana without lexical normalisation.",
    version="0.1.0",
)

LOGGER = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_origins=["null"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Provide basic navigation links for manual exploration."""

    return {"docs": "/docs", "openapi": "/openapi.json"}


@app.on_event("startup")
def initialise_pipeline() -> None:
    """Prepare heavyweight resources on startup."""

    pipeline = get_pipeline()
    pipeline.warm_up()
    LOGGER.info("backend=%s", pipeline.backend_name())


@app.get("/healthz", summary="Health check")
def health_check(pipeline: PhonemePipeline = Depends(get_pipeline)) -> dict[str, str]:
    """Expose the current transcription backend."""

    return {"backend": pipeline.backend_name()}


@app.post(
    "/transcribe_phonetic",
    response_model=PhoneticTranscriptionResponse,
    summary="Decode speech audio into katakana",
)
async def transcribe_phonetic(
    audio: UploadFile = File(..., description="16kHz mono WAV input"),
    style: KanaStyle = Form(default=KanaStyle.rough, description="Kana rendering style"),
    final_c_t: FinalConsonantHandling = Form(
        default=FinalConsonantHandling.xtsu,
        description="Trailing stop consonant handling",
    ),
    th: THMapping = Form(default=THMapping.su, description="TH mapping preference"),
    pipeline: PhonemePipeline = Depends(get_pipeline),
) -> PhoneticTranscriptionResponse:
    """Endpoint stub; wires options and delegates to the phoneme pipeline."""

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    try:
        # Pipeline handles ONLY raw audio -> phonemes. Do not pass rendering flags here.
        decoded = await pipeline.transcribe(payload)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    enriched = _apply_kana(
        decoded,
        style=style,
        final_c_t=final_c_t,
        th=th,
    )
    return _to_response(enriched)


@app.post(
    "/transcribe_phonetic_any",
    response_model=PhoneticTranscriptionResponse,
    summary="Decode speech audio in multiple formats into katakana",
)
async def transcribe_phonetic_any(
    audio: UploadFile = File(..., description="Audio stream in common browser formats"),
    style: KanaStyle = Form(default=KanaStyle.rough, description="Kana rendering style"),
    final_c_t: FinalConsonantHandling = Form(
        default=FinalConsonantHandling.xtsu,
        description="Trailing stop consonant handling",
    ),
    th: THMapping = Form(default=THMapping.su, description="TH mapping preference"),
    pipeline: PhonemePipeline = Depends(get_pipeline),
) -> PhoneticTranscriptionResponse:
    """Accept arbitrary audio containers, normalise to WAV, and delegate to the pipeline."""

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    try:
        wav_bytes = await asyncio.to_thread(
            _transcode_audio_to_wav,
            payload,
            filename=audio.filename,
            content_type=audio.content_type,
        )
    except TranscodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        decoded = await pipeline.transcribe(wav_bytes)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    enriched = _apply_kana(
        decoded,
        style=style,
        final_c_t=final_c_t,
        th=th,
    )
    return _to_response(enriched)


def _apply_kana(
    result: PhonemeResult,
    *,
    style: KanaStyle,
    final_c_t: FinalConsonantHandling,
    th: THMapping,
) -> PhonemeResult:
    kana_values = to_kana_sequence(
        (phone.symbol for phone in result.phones),
        kana_style=style.value,
        final_c_t=final_c_t.value,
        th_style=th.value,
    )
    kana_tokens = [
        KanaToken(value=value, start=phone.start, end=phone.end)
        for phone, value in zip(result.phones, kana_values, strict=True)
    ]
    return PhonemeResult(phones=result.phones, kana=kana_tokens)


def _to_response(result: PhonemeResult) -> PhoneticTranscriptionResponse:
    phones = [
        PhonemeSpan(
            symbol=phone.symbol,
            start=phone.start,
            end=phone.end,
            confidence=phone.confidence,
        )
        for phone in result.phones
    ]
    kana = [
        KanaSpan(kana=token.value, start=token.start, end=token.end)
        for token in (result.kana or ())
    ]
    return PhoneticTranscriptionResponse(phones=phones, kana=kana, kana_text=result.kana_text)


def _guess_extension(filename: str | None, content_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            return guessed
    return ".bin"


class TranscodeError(RuntimeError):
    """Raised when audio transcoding fails."""


def _transcode_audio_to_wav(
    data: bytes,
    *,
    filename: str | None,
    content_type: str | None,
) -> bytes:
    """Persist bytes, invoke ffmpeg, and load the resulting WAV payload."""

    source_suffix = _guess_extension(filename, content_type)
    source_file = tempfile.NamedTemporaryFile(delete=False, suffix=source_suffix)
    target_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    source_path = Path(source_file.name)
    target_path = Path(target_file.name)
    try:
        try:
            source_file.write(data)
            source_file.flush()
        finally:
            source_file.close()
        target_file.close()
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(target_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise TranscodeError("ffmpeg not found") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
            detail = "conversion failed"
            if stderr:
                detail = f"{detail}: {stderr}"
            raise TranscodeError(detail) from exc
        return target_path.read_bytes()
    finally:
        for temp_path in (source_path, target_path):
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
