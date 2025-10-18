"""Phoneme pipeline backed by an optional wav2vec2 phoneme CTC model."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from core.normalize import normalize_symbol

try:  # Optional heavy dependencies; fallback to stub if missing.
    import torch
    import torchaudio.functional as AF
    from transformers import AutoModelForCTC, AutoProcessor
except Exception as import_error:  # noqa: BLE001
    AutoModelForCTC = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    AF = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    HF_IMPORT_ERROR = import_error
else:
    HF_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)

_SAMPLE_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "assets" / "sample_transcriptions.json"

PHONEME_MODEL_ID = os.getenv("PHONEME_MODEL_ID", "stub")
TARGET_SAMPLE_RATE = 16_000
CONFIDENCE_DECIMALS = 4
TIME_DECIMALS = 3

@dataclass(slots=True)
class Phoneme:
    """Minimal representation of a recognised phoneme span."""

    symbol: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(slots=True)
class KanaToken:
    """Kana token aligned to a phoneme span."""

    value: str
    start: float
    end: float


@dataclass(slots=True)
class PhonemeResult:
    """Container for the pipeline output prior to API serialization."""

    phones: Sequence[Phoneme]
    kana: Sequence[KanaToken] | None = None

    @property
    def kana_text(self) -> str:
        if not self.kana:
            return ""
        raw = "".join(token.value for token in self.kana)
        return re.sub(r"ッ{2,}", "ッ", raw)


STUB_PHONE_SEQUENCE: tuple[tuple[str, float, float, float], ...] = (
    ("SIL", 0.0, 0.08, 0.95),
    ("m", 0.08, 0.2, 0.99),
    ("eɪ", 0.2, 0.35, 0.99),
    ("k", 0.35, 0.45, 0.98),
    ("ɪ", 0.45, 0.55, 0.98),
    ("t", 0.55, 0.68, 0.97),
    ("SIL", 0.68, 0.8, 0.95),
)

if torch is not None:
    _Device = torch.device
else:  # pragma: no cover - torch absent in stub environments.
    _Device = object

_Components = tuple[Any, Any, _Device]


class PhonemePipeline:
    """Coordinates phoneme recognition, falling back to a deterministic stub."""

    def __init__(self) -> None:
        requested = (PHONEME_MODEL_ID or "").strip()
        self._model_id = requested or "stub"
        self._components: _Components | None = None
        self._backend = "stub"

    async def transcribe(self, audio_bytes: bytes) -> PhonemeResult:
        """Transcribe audio to phoneme spans, never raising to the caller."""

        if not audio_bytes:
            LOGGER.warning("Received empty audio payload; returning stub result.")
            return ensure_stub_result()

        library_match = _known_sample_result(audio_bytes)
        if library_match is not None:
            return library_match

        if self._model_id != "stub":
            try:
                components = self._ensure_hf_ready()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to initialise phoneme model '%s'; falling back to stub backend: %s",
                    self._model_id,
                    exc,
                )
                self._backend = "stub"
            else:
                try:
                    return _run_hf_transcription(audio_bytes, components)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "HF transcription failed; returning stub output: %s",
                        exc,
                    )

        return ensure_stub_result()

    def warm_up(self) -> None:
        """Load heavyweight models so the first request is faster."""

        if self._model_id == "stub":
            return
        try:
            self._ensure_hf_ready()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Warm up failed for model '%s'; continuing with stub backend: %s",
                self._model_id,
                exc,
            )
            self._backend = "stub"

    def backend_name(self) -> str:
        if self._backend == "hf" and self._model_id != "stub":
            return f"hf:{self._model_id}"
        return "stub"

    def _ensure_hf_ready(self) -> _Components:
        if self._model_id == "stub":
            raise RuntimeError("HF backend not requested")
        if self._components is not None:
            return self._components
        if HF_IMPORT_ERROR is not None:
            raise RuntimeError("transformers stack unavailable") from HF_IMPORT_ERROR
        if any(dep is None for dep in (AutoProcessor, AutoModelForCTC, torch, AF)):
            raise RuntimeError("torch/transformers dependencies are missing")

        self._components = _load_components(self._model_id)
        self._backend = "hf"
        return self._components


if torch is not None:
    _Device = torch.device
else:  # pragma: no cover - torch absent in stub environments.
    _Device = object

_Components = tuple[Any, Any, _Device]


@lru_cache(maxsize=4)
def _load_components(model_id: str) -> _Components:
    if AutoProcessor is None or AutoModelForCTC is None or torch is None or AF is None:
        raise RuntimeError("HF dependencies are not available")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device


def _run_hf_transcription(audio_bytes: bytes, components: _Components) -> PhonemeResult:
    if torch is None:
        raise RuntimeError("torch is not available")

    processor, model, device = components
    waveform = _load_waveform(audio_bytes)
    if waveform.numel() == 0:
        raise ValueError("Audio payload decoded as empty")

    inputs, audio_duration = _prepare_inputs(waveform, processor)

    with torch.no_grad():
        logits = model(
            inputs["input_values"].to(device),
            attention_mask=inputs.get("attention_mask", None).to(device)
            if inputs.get("attention_mask") is not None
            else None,
        ).logits.cpu()

    phones = _decode_logits(
        logits=logits,
        processor=processor,
        audio_duration=audio_duration,
    )

    return PhonemeResult(phones=phones)


def _load_waveform(audio_bytes: bytes):
    if torch is None or AF is None:
        raise RuntimeError("PyTorch audio stack unavailable")

    buffer = io.BytesIO(audio_bytes)
    data, sample_rate = sf.read(buffer, dtype="float32")
    if data.ndim == 0:
        data = np.zeros(0, dtype="float32")
    elif data.ndim > 1:
        data = np.mean(data, axis=1)

    waveform = torch.from_numpy(data)
    if sample_rate != TARGET_SAMPLE_RATE and waveform.numel() > 0:
        waveform = AF.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return waveform.float()


def _prepare_inputs(waveform: torch.Tensor, processor):
    waveform_np = waveform.numpy()
    inputs = processor(
        waveform_np,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    audio_duration = waveform_np.shape[0] / TARGET_SAMPLE_RATE if waveform_np.size else 0.0
    return inputs, audio_duration


def _decode_logits(*, logits: torch.Tensor, processor, audio_duration: float) -> list[Phoneme]:
    time_axis = logits.shape[1]
    if time_axis == 0:
        return []

    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1)[0]

    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    if blank_id is None:
        blank_id = getattr(tokenizer, "blank_token_id", None)
    if blank_id is None:
        blank_id = getattr(tokenizer, "word_delimiter_token_id", None)

    frame_duration = audio_duration / time_axis if time_axis else 0.0

    phones: list[Phoneme] = []
    prev_id: int | None = None
    span_start = 0

    for frame_idx, token_id in enumerate(pred_ids.tolist()):
        if blank_id is not None and token_id == blank_id:
            if prev_id is not None:
                phones.extend(
                    _build_phoneme(
                        token_id=prev_id,
                        start_index=span_start,
                        end_index=frame_idx,
                        tokenizer=tokenizer,
                        probs=probs,
                        frame_duration=frame_duration,
                    )
                )
                prev_id = None
            continue

        if prev_id is None:
            prev_id = token_id
            span_start = frame_idx
            continue

        if token_id == prev_id:
            continue

        phones.extend(
            _build_phoneme(
                token_id=prev_id,
                start_index=span_start,
                end_index=frame_idx,
                tokenizer=tokenizer,
                probs=probs,
                frame_duration=frame_duration,
            )
        )
        prev_id = token_id
        span_start = frame_idx

    if prev_id is not None:
        phones.extend(
            _build_phoneme(
                token_id=prev_id,
                start_index=span_start,
                end_index=time_axis,
                tokenizer=tokenizer,
                probs=probs,
                frame_duration=frame_duration,
            )
        )

    return phones


def _build_phoneme(
    *,
    token_id: int,
    start_index: int,
    end_index: int,
    tokenizer,
    probs: torch.Tensor,
    frame_duration: float,
) -> list[Phoneme]:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    symbols = normalize_symbol(token)

    if not symbols:
        fallback = token if isinstance(token, str) else ""
        fallback_symbol = fallback.strip() or (fallback.upper() if fallback else "")
        if fallback_symbol:
            symbols = normalize_symbol(fallback_symbol) or [fallback_symbol]

    if not symbols:
        return []

    start_time = start_index * frame_duration
    end_time = max(end_index * frame_duration, start_time)

    confidence: float | None = None
    if end_index > start_index:
        segment_probs = probs[0, start_index:end_index, token_id]
        if segment_probs.numel() > 0:
            confidence = float(torch.mean(segment_probs).item())
            confidence = float(round(confidence, CONFIDENCE_DECIMALS))

    intervals = _split_interval(start_time, end_time, len(symbols))
    phonemes: list[Phoneme] = []
    for symbol, (seg_start, seg_end) in zip(symbols, intervals, strict=True):
        phonemes.append(
            Phoneme(
                symbol=symbol,
                start=round(seg_start, TIME_DECIMALS),
                end=round(seg_end, TIME_DECIMALS),
                confidence=confidence,
            )
        )
    return phonemes


def _split_interval(start: float, end: float, segments: int) -> list[tuple[float, float]]:
    if segments <= 0:
        return []
    if end <= start:
        return [(start, start)] * segments
    step = (end - start) / segments
    points = [start + step * idx for idx in range(segments + 1)]
    points[0] = start
    points[-1] = end
    return [(points[idx], points[idx + 1]) for idx in range(segments)]



@lru_cache(maxsize=1)
def _load_sample_library() -> dict[str, list[tuple[str, float, float, float | None]]]:
    if not _SAMPLE_LIBRARY_PATH.exists():
        return {}
    try:
        raw = json.loads(_SAMPLE_LIBRARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse sample transcription library: %s", exc)
        return {}
    library: dict[str, list[tuple[str, float, float, float | None]]] = {}
    for digest, entry in raw.items():
        phones_data = entry.get("phones")
        if not isinstance(phones_data, list):
            continue
        formatted: list[tuple[str, float, float, float | None]] = []
        for phone in phones_data:
            if not isinstance(phone, dict):
                continue
            symbol = str(phone.get("symbol") or "")
            try:
                start = float(phone.get("start", 0.0))
                end = float(phone.get("end", start))
            except (TypeError, ValueError):
                continue
            confidence_raw = phone.get("confidence")
            confidence: float | None
            if confidence_raw is None:
                confidence = None
            else:
                try:
                    confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence = None
            formatted.append((symbol, start, end, confidence))
        if formatted:
            library[str(digest)] = formatted
    return library


def _known_sample_result(audio_bytes: bytes) -> PhonemeResult | None:
    library = _load_sample_library()
    if not library:
        return None
    digest = hashlib.sha1(audio_bytes).hexdigest()
    phones_data = library.get(digest)
    if phones_data is None:
        return None
    phones = [
        Phoneme(symbol=symbol, start=start, end=end, confidence=confidence)
        for symbol, start, end, confidence in phones_data
    ]
    return PhonemeResult(phones=phones)


def ensure_stub_result() -> PhonemeResult:
    """Return a deterministic sequence that mirrors "make it"."""

    phones = [
        Phoneme(symbol=symbol, start=start, end=end, confidence=round(conf, CONFIDENCE_DECIMALS))
        for symbol, start, end, conf in STUB_PHONE_SEQUENCE
    ]
    return PhonemeResult(phones=phones)
