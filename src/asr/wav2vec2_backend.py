"""wav2vec2 CTC を用いた音素推定パイプライン。"""
from __future__ import annotations

import threading
import time
import torch
from transformers import AutoModelForCTC, AutoProcessor

from .audio import ensure_sample_rate, load_waveform
from .pipeline import PhonemePipeline, PhonemeSpan, TranscriptionMetrics
from phones.decode import build_phoneme_spans, register_processor

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
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device_spec = device
        self._chunk_ms = chunk_ms
        self._overlap = overlap
        self._conf_threshold = conf_threshold
        self._min_phone_ms = min_phone_ms
        self._components: tuple[AutoProcessor, AutoModelForCTC, torch.device] | None = None
        self._cold_start_ms: float | None = None
        self._load_lock = threading.Lock()

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

        processor, model, device = self.ensure_ready()
        blank_id = processor.tokenizer.pad_token_id
        if blank_id is None and getattr(processor.tokenizer, 'pad_token', None) is not None:
            blank_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
        if blank_id is None and getattr(processor.tokenizer, 'eos_token_id', None) is not None:
            blank_id = processor.tokenizer.eos_token_id
        if blank_id is None:
            raise RuntimeError('blank token id could not be determined')

        chunk_samples = max(1, int(sample_rate * (self._chunk_ms / 1000.0)))
        hop_samples = max(1, int(chunk_samples * (1.0 - self._overlap)))
        min_duration = self._min_phone_ms / 1000.0
        total_samples = waveform.shape[-1]
        all_spans: list[tuple[str, float, float, float | None]] = []

        infer_start = time.perf_counter()
        start_sample = 0
        chunk_index = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[start_sample:end_sample]
            actual_samples = chunk.shape[-1]
            if actual_samples == 0:
                break

            chunk_np = chunk.unsqueeze(0).numpy()
            inputs = processor(
                chunk_np,
                sampling_rate=sample_rate,
                return_tensors='pt',
                padding=True,
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

            chunk_duration = actual_samples / sample_rate
            frame_duration = chunk_duration / frame_count if frame_count else 0.0
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
            conf_threshold=self._conf_threshold,
            min_phone_ms=self._min_phone_ms,
        )
        return phones, metrics

    def _ensure_ready(self) -> tuple[AutoProcessor, AutoModelForCTC, torch.device]:
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
            processor = (
                AutoProcessor.from_pretrained(self._model_id, revision=revision)
                if use_revision
                else AutoProcessor.from_pretrained(self._model_id)
            )
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

    def ensure_ready(self) -> tuple[AutoProcessor, AutoModelForCTC, torch.device]:
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

