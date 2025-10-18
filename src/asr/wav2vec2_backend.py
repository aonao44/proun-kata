"""wav2vec2 CTC を用いた音素推定パイプライン。"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.normalize import ALLOWED_SYMBOLS, normalize_symbol
from .audio import ensure_sample_rate, load_waveform
from .ctc_decode import (
    BeamSearchConfig,
    PhoneLanguageModel,
    TokenCatalog,
    beam_search_ctc,
    build_token_catalog,
    load_phone_language_model,
)
from .postprocess import apply_phonetic_postprocessing
from .pipeline import PhonemePipeline, PhonemeSpan, ShortAudioError, TranscriptionMetrics
from phones.decode import build_phoneme_spans, register_processor

LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("invalid %s=%s; using default %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("invalid %s=%s; using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    LOGGER.warning("invalid %s=%s; using default %s", name, raw, default)
    return default


def _load_processor_with_fallback(
    model_id: str,
    *,
    revision: str | None,
    tokenizer_id: str | None,
) -> Wav2Vec2Processor:
    revision_kwargs = {"revision": revision} if revision else {}
    try:
        return AutoProcessor.from_pretrained(model_id, **revision_kwargs)
    except Exception as exc:  # pragma: no cover - logging branch
        LOGGER.warning(
            "AutoProcessor load failed model_id=%s revision=%s: %s",
            model_id,
            revision or "<default>",
            exc,
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **revision_kwargs)
    tokenizer_source = tokenizer_id or model_id
    tokenizer_kwargs = {}
    if tokenizer_id is None and revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer: PreTrainedTokenizerBase | None = None
    auto_exc: Exception | None = None
    try:
        candidate = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    except Exception as tokenizer_exc:
        auto_exc = tokenizer_exc
    else:
        if isinstance(candidate, PreTrainedTokenizerBase):
            tokenizer = candidate
        else:  # pragma: no cover - logging branch
            LOGGER.warning(
                "AutoTokenizer returned unexpected type %s for %s; falling back to Wav2Vec2CTCTokenizer",
                type(candidate).__name__,
                tokenizer_source,
            )

    if tokenizer is None:
        try:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        except Exception as ctc_exc:
            message = (
                "Tokenizer load failed for "
                f"'{tokenizer_source}' (revision={revision or '<default>'}). "
                "Set HF_TOKENIZER_ID or use a repository that bundles a tokenizer."
            )
            details: list[str] = []
            if auto_exc is not None:
                details.append(f"AutoTokenizer error: {auto_exc}")
            details.append(f"CTC tokenizer error: {ctc_exc}")
            if details:
                message = f"{message} {' '.join(details)}"
            raise RuntimeError(message) from ctc_exc

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise RuntimeError(
            "Tokenizer load returned an unexpected object type "
            f"({type(tokenizer).__name__}). Set HF_TOKENIZER_ID or use a repository that "
            "bundles a compatible tokenizer."
        )


    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _validate_english_vocab(tokenizer: PreTrainedTokenizerBase) -> None:
    try:
        raw_vocab = tokenizer.get_vocab()
    except Exception as exc:  # pragma: no cover - logging branch
        LOGGER.warning("failed to read tokenizer vocab: %s", exc)
        return

    tokens: set[str] = set()
    if isinstance(raw_vocab, dict):
        tokens.update(str(key) for key in raw_vocab.keys())
    else:
        tokens.update(str(entry) for entry in raw_vocab)

    if hasattr(tokenizer, "get_added_vocab"):
        try:
            added_vocab = tokenizer.get_added_vocab()
        except Exception:  # pragma: no cover - logging branch
            added_vocab = {}
        if isinstance(added_vocab, dict):
            tokens.update(str(key) for key in added_vocab.keys())

    invalid: dict[str, list[str]] = {}
    normalized_seen: set[str] = set()
    for token in sorted(tokens):
        normalized = normalize_symbol(token, log_unknown=False)
        if not normalized:
            continue
        normalized_seen.update(normalized)
        outside = [symbol for symbol in normalized if symbol not in ALLOWED_SYMBOLS]
        if outside:
            invalid[token] = outside

    if invalid:
        for token, symbols in invalid.items():
            LOGGER.error("english_vocab_mismatch token=%s normalized=%s", token, symbols)
        offenders = sorted({symbol for symbols in invalid.values() for symbol in symbols})
        raise RuntimeError(
            "Tokenizer vocabulary contains non-English phonemes: " + ", ".join(offenders)
        )

    LOGGER.info(
        "english_vocab_validation_ok tokens=%d normalized=%d",
        len(tokens),
        len(normalized_seen),
    )


class Wav2Vec2Pipeline(PhonemePipeline):
    """指定モデルを用いて音素列を生成する。"""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        chunk_ms: int,
        overlap: float,
        conf_threshold: float,
        min_phone_ms: float,
        min_input_ms: float,
        reject_ms: float,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device_spec = device
        self._chunk_ms = chunk_ms
        self._overlap = overlap
        self._conf_threshold = conf_threshold
        self._min_phone_ms = min_phone_ms
        self._min_input_ms = max(0.0, float(min_input_ms))
        self._reject_ms = max(0.0, float(reject_ms))
        default_lm_path = Path(__file__).resolve().parents[2] / "assets" / "phonelm" / "phone_bigram.json"
        self._lm_path = os.getenv("DECODE_LM_PATH", str(default_lm_path))
        self._beam_config = BeamSearchConfig(
            beam_size=max(1, _env_int("DECODE_BEAM_SIZE", 12)),
            vowel_bonus=_env_float("DECODE_VOWEL_BONUS", 0.15),
            repeat_penalty=_env_float("DECODE_REPEAT_PENALTY", 0.2),
            final_cons_penalty=_env_float("DECODE_FINAL_CONS_PENALTY", 0.3),
            th_bonus=_env_float("DECODE_TH_BONUS", 0.0),
            use_voicing=_env_bool("DECODE_USE_VOICING", False),
            voicing_bonus=0.2,
            voicing_threshold=0.4,
            lm_weight=_env_float("DECODE_LM_WEIGHT", 0.0),
            insertion_penalty=_env_float("DECODE_INSERTION_PENALTY", 0.0),
        )
        self._phone_lm: PhoneLanguageModel | None = None
        self._token_catalog: TokenCatalog | None = None
        self._components: tuple[Wav2Vec2Processor, AutoModelForCTC, torch.device] | None = None
        self._cold_start_ms: float | None = None
        self._load_lock = threading.Lock()
        LOGGER.info(
            "decoder_config beam=%d vowel=%.2f repeat=%.2f final=%.2f th=%.2f voicing=%s",
            self._beam_config.beam_size,
            self._beam_config.vowel_bonus,
            self._beam_config.repeat_penalty,
            self._beam_config.final_cons_penalty,
            self._beam_config.th_bonus,
            self._beam_config.use_voicing,
        )

    async def transcribe(self, audio_bytes: bytes, *, req_id: str) -> tuple[list[PhonemeSpan], TranscriptionMetrics]:
        if not audio_bytes:
            raise ValueError('audio payload is empty')

        total_start = time.perf_counter()
        waveform, sample_rate = load_waveform(audio_bytes)
        duration = waveform.shape[-1] / float(sample_rate)
        if duration <= 0.0:
            raise ValueError('decoded audio contains no samples')
        if duration > 30.0:
            raise ValueError('audio exceeds 30 seconds limit')

        waveform, sample_rate = ensure_sample_rate(waveform, sample_rate)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.to(torch.float32)
        peak = waveform.abs().max()
        if torch.isfinite(peak) and peak > 0:
            waveform = waveform / peak
        waveform = waveform.clamp(-1.0, 1.0).contiguous()

        original_samples = waveform.shape[-1]
        original_ms = original_samples / float(sample_rate) * 1000.0
        LOGGER.debug("input_len=%d samples (%.2f ms)", original_samples, original_ms)

        if self._reject_ms and original_ms < self._reject_ms:
            raise ShortAudioError(original_ms, self._reject_ms)

        min_input_samples = int(round(sample_rate * (self._min_input_ms / 1000.0)))
        if min_input_samples > 0 and waveform.shape[-1] < min_input_samples:
            pad_amount = min_input_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_amount))

        processor, model, device = self.ensure_ready()
        blank_id = processor.tokenizer.pad_token_id
        if blank_id is None and getattr(processor.tokenizer, 'pad_token', None) is not None:
            blank_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
        if blank_id is None and getattr(processor.tokenizer, 'eos_token_id', None) is not None:
            blank_id = processor.tokenizer.eos_token_id
        if blank_id is None:
            raise RuntimeError('blank token id could not be determined')

        catalog = self._token_catalog
        lm_instance = self._phone_lm
        if catalog is None or lm_instance is None:
            raise RuntimeError('decoder configuration is not initialised')

        chunk_samples = max(1, int(sample_rate * (self._chunk_ms / 1000.0)))
        hop_samples = max(1, int(chunk_samples * (1.0 - self._overlap)))
        min_duration = self._min_phone_ms / 1000.0
        total_samples = waveform.shape[-1]
        all_spans: list[tuple[str, float, float, float | None]] = []
        decoded_chunks: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []
        effective_conf_threshold = self._conf_threshold
        effective_min_phone_ms = self._min_phone_ms

        infer_start = time.perf_counter()
        start_sample = 0
        chunk_index = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[start_sample:end_sample]
            if chunk.shape[-1] == 0:
                break

            chunk, original_samples = self._prepare_for_model(chunk, sample_rate)
            chunk_np = chunk.unsqueeze(0).numpy()
            inputs = processor(
                chunk_np,
                sampling_rate=sample_rate,
                return_tensors='pt',
                padding=True,
                return_attention_mask=True,
            )
            input_values = inputs['input_values'].to(device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.inference_mode():
                logits = model(input_values=input_values, attention_mask=attention_mask).logits[0]

            probs = torch.softmax(logits, dim=-1).cpu()
            frame_count = probs.shape[0]
            if frame_count == 0:
                start_sample += hop_samples
                chunk_index += 1
                continue

            chunk_duration = original_samples / sample_rate if original_samples else 0.0
            frame_duration = chunk_duration / frame_count if frame_count else 0.0
            try:
                decoder_result = beam_search_ctc(
                    logits,
                    blank_id=blank_id,
                    catalog=catalog,
                    lm=lm_instance,
                    config=self._beam_config,
                )
                token_ids = torch.tensor(decoder_result.alignment, dtype=torch.long)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("beam_search_failed; falling back to argmax: %s", exc)
                token_ids = torch.argmax(probs, dim=-1)

            trim_start_frames = 0
            if chunk_index > 0 and self._overlap > 0.0:
                overlap_frames = int(round(frame_count * self._overlap))
                if overlap_frames >= frame_count:
                    overlap_frames = frame_count - 1
                trim_start_frames = max(0, overlap_frames)
            if trim_start_frames:
                token_ids = token_ids[trim_start_frames:]
                probs = probs[trim_start_frames:]
                frame_count = token_ids.shape[0]
            if frame_count == 0:
                start_sample += hop_samples
                chunk_index += 1
                continue

            offset_time = (start_sample / sample_rate) + (trim_start_frames * frame_duration)
            decoded_chunks.append((token_ids, probs, frame_duration, offset_time))
            spans = build_phoneme_spans(
                token_ids,
                probabilities=probs,
                blank_id=blank_id,
                frame_duration=frame_duration,
                conf_threshold=self._conf_threshold,
                min_duration=min_duration,
            )
            for symbol, start_time, end_time, confidence in spans:
                all_spans.append((symbol, start_time + offset_time, end_time + offset_time, confidence))

            start_sample += hop_samples
            chunk_index += 1

        infer_end = time.perf_counter()
        merged_spans = _merge_adjacent_spans(all_spans)
        phones = [
            PhonemeSpan(
                symbol=symbol,
                start=round(start, 3),
                end=round(end, 3),
                confidence=confidence,
            )
            for symbol, start, end, confidence in merged_spans
        ]

        if not phones and decoded_chunks:
            LOGGER.debug(
                "rescue_decode: conf_threshold=%s min_phone_ms=%s",
                self._conf_threshold,
                self._min_phone_ms,
            )
            all_spans = []
            for token_ids, probs, frame_duration, offset_time in decoded_chunks:
                spans = build_phoneme_spans(
                    token_ids,
                    probabilities=probs,
                    blank_id=blank_id,
                    frame_duration=frame_duration,
                    conf_threshold=0.0,
                    min_duration=0.0,
                )
                for symbol, start_time, end_time, confidence in spans:
                    all_spans.append(
                        (symbol, start_time + offset_time, end_time + offset_time, confidence)
                    )

            merged_spans = _merge_adjacent_spans(all_spans)
            phones = [
                PhonemeSpan(
                    symbol=symbol,
                    start=round(start, 3),
                    end=round(end, 3),
                    confidence=confidence,
                )
                for symbol, start, end, confidence in merged_spans
            ]
            effective_conf_threshold = 0.0
            effective_min_phone_ms = 0

        if phones:
            phones = apply_phonetic_postprocessing(phones)

        total_end = time.perf_counter()
        metrics = TranscriptionMetrics(
            req_id=req_id,
            model_id=self._model_id,
            revision=self._revision,
            device=str(device),
            cold_start_ms=self._cold_start_ms,
            inference_ms=(infer_end - infer_start) * 1000,
            total_ms=(total_end - total_start) * 1000,
            chunk_ms=self._chunk_ms,
            overlap=self._overlap,
            conf_threshold=effective_conf_threshold,
            min_phone_ms=effective_min_phone_ms,
            min_input_ms=self._min_input_ms,
            reject_ms=self._reject_ms,
        )
        return phones, metrics

    def _ensure_ready(self) -> tuple[Wav2Vec2Processor, AutoModelForCTC, torch.device]:
        if self._components is not None:
            return self._components

        with self._load_lock:
            if self._components is not None:
                return self._components

            device = self._resolve_device()
            load_start_cpu = time.perf_counter()
            load_event_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            load_event_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if load_event_start is not None and load_event_end is not None:
                load_event_start.record()

            revision = (self._revision or "").strip()
            use_revision = bool(revision and revision.upper() != "REPLACE_WITH_COMMIT_SHA")
            resolved_revision = revision if use_revision else None
            tokenizer_id = os.getenv("HF_TOKENIZER_ID")
            if tokenizer_id is not None:
                tokenizer_id = tokenizer_id.strip() or None

            log_revision = resolved_revision or "<default>"
            log_tokenizer = tokenizer_id or self._model_id
            LOGGER.info(
                "loading wav2vec2 processor model_id=%s revision=%s tokenizer_id=%s",
                self._model_id,
                log_revision,
                log_tokenizer,
            )

            processor = _load_processor_with_fallback(
                self._model_id,
                revision=resolved_revision,
                tokenizer_id=tokenizer_id,
            )
            _validate_english_vocab(processor.tokenizer)
            feature_extractor = getattr(processor, "feature_extractor", None)
            if feature_extractor is not None:
                sample_rate = getattr(feature_extractor, "sampling_rate", 16000)
                chunk_length = int(sample_rate * (self._chunk_ms / 1000.0))
                stride_length = int(chunk_length * (1.0 - self._overlap))
                if hasattr(feature_extractor, "chunk_length"):
                    feature_extractor.chunk_length = max(chunk_length, 1)
                if hasattr(feature_extractor, "stride_length"):
                    feature_extractor.stride_length = max(stride_length, 1)

            model = (
                AutoModelForCTC.from_pretrained(self._model_id, revision=revision)
                if use_revision
                else AutoModelForCTC.from_pretrained(self._model_id)
            )
            register_processor(processor)
            if self._token_catalog is None:
                self._token_catalog = build_token_catalog(processor.tokenizer)
            if self._phone_lm is None:
                symbols = {symbol for values in self._token_catalog.id_to_canonical.values() for symbol in values}
                self._phone_lm = load_phone_language_model(self._lm_path, symbols)
            model.to(device)
            model.eval()

            load_end_cpu = time.perf_counter()
            if load_event_start is not None and load_event_end is not None:
                load_event_end.record()
                torch.cuda.synchronize()
                self._cold_start_ms = load_event_start.elapsed_time(load_event_end)
            else:
                self._cold_start_ms = (load_end_cpu - load_start_cpu) * 1000

            self._components = (processor, model, device)

        return self._components

    def _prepare_for_model(self, chunk: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        original_samples = chunk.shape[-1]
        duration_ms = original_samples / float(sample_rate) * 1000.0

        min_samples = int(round(sample_rate * (self._min_input_ms / 1000.0)))
        pad = max(0, min_samples - original_samples)
        if pad > 0:
            chunk = F.pad(chunk, (0, pad))
            LOGGER.debug(
                "input_len=%d samples (%.2f ms) padded=+%d",
                original_samples,
                duration_ms,
                pad,
            )
        else:
            LOGGER.debug(
                "input_len=%d samples (%.2f ms)",
                original_samples,
                duration_ms,
            )
        return chunk, original_samples

    def ensure_ready(self) -> tuple[Wav2Vec2Processor, AutoModelForCTC, torch.device]:
        """Public wrapper to keep call sites stable."""

        return self._ensure_ready()

    def _resolve_device(self) -> torch.device:
        if self._device_spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self._device_spec)


def _merge_adjacent_spans(spans: list[tuple[str, float, float, float | None]]) -> list[tuple[str, float, float, float | None]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: item[1])
    merged: list[tuple[str, float, float, float | None]] = []
    for symbol, start, end, confidence in ordered:
        if not merged:
            merged.append((symbol, start, end, confidence))
            continue
        last_symbol, last_start, last_end, last_conf = merged[-1]
        if symbol == last_symbol and start <= last_end + 1e-3:
            duration_prev = last_end - last_start
            duration_new = end - start
            merged_conf = _combine_confidence(last_conf, confidence, duration_prev, duration_new)
            merged[-1] = (
                last_symbol,
                last_start,
                max(last_end, end),
                merged_conf,
            )
        else:
            merged.append((symbol, start, end, confidence))
    return merged


def _combine_confidence(
    first: float | None,
    second: float | None,
    first_duration: float,
    second_duration: float,
) -> float | None:
    if first is None and second is None:
        return None
    weight_first = first_duration if first is not None else 0.0
    weight_second = second_duration if second is not None else 0.0
    total_weight = weight_first + weight_second
    if total_weight <= 0:
        return second if second is not None else first
    value = 0.0
    if first is not None:
        value += first * weight_first
    if second is not None:
        value += second * weight_second
    return round(value / total_weight, 4)
