"""CTC生ログから音素スパンを抽出する処理。"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .normalise import normalize_symbol


def collapse_repeats(ids: Iterable[int], *, blank_id: int) -> list[int]:
    collapsed: list[int] = []
    prev_id: int | None = None
    for idx in ids:
        if idx == blank_id:
            prev_id = None
            continue
        if idx == prev_id:
            continue
        collapsed.append(idx)
        prev_id = idx
    return collapsed


def build_phoneme_spans(
    token_ids: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    blank_id: int,
    frame_duration: float,
    conf_threshold: float,
    min_duration: float,
) -> list[tuple[str, float, float, float | None]]:
    spans: list[tuple[str, float, float, float | None]] = []
    current_id: int | None = None
    start_index = 0

    for index, token_id in enumerate(token_ids.tolist()):
        if token_id == blank_id:
            if current_id is not None:
                _finalise_span(
                    spans,
                    current_id,
                    start_index,
                    index,
                    probabilities,
                    frame_duration,
                    conf_threshold,
                    min_duration,
                )
                current_id = None
            continue
        if token_id == current_id:
            continue
        if current_id is not None:
            _finalise_span(
                spans,
                current_id,
                start_index,
                index,
                probabilities,
                frame_duration,
                conf_threshold,
                min_duration,
            )
        current_id = token_id
        start_index = index

    if current_id is not None:
        _finalise_span(
            spans,
            current_id,
            start_index,
            token_ids.shape[0],
            probabilities,
            frame_duration,
            conf_threshold,
            min_duration,
        )
    return [span for span in spans if span[0]]


def _finalise_span(
    spans: list[tuple[str, float, float, float | None]],
    token_id: int,
    start_index: int,
    end_index: int,
    probabilities: torch.Tensor,
    frame_duration: float,
    conf_threshold: float,
    min_duration: float,
) -> None:
    processor = _lazy_processor()
    raw_token = processor.tokenizer.convert_ids_to_tokens(token_id)
    symbols = normalize_symbol(raw_token)
    if not symbols:
        return

    silence_symbols = {"SIL", "SP", "NSN"}
    total_start = start_index * frame_duration
    total_end = end_index * frame_duration
    segment = probabilities[start_index:end_index, token_id]
    confidence: float | None = None
    if segment.numel() > 0:
        confidence = float(torch.mean(segment).item())
        if confidence < conf_threshold and not all(sym in silence_symbols for sym in symbols):
            return

    intervals = _split_interval(total_start, total_end, len(symbols))
    effective_min = min_duration if len(symbols) == 1 else 0.0

    for symbol, (seg_start, seg_end) in zip(symbols, intervals, strict=True):
        seg_duration = seg_end - seg_start if seg_end >= seg_start else 0.0
        if symbol not in silence_symbols and seg_duration < effective_min:
            continue
        spans.append((symbol, round(seg_start, 3), round(seg_end, 3), confidence))


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



_lazy_cache: dict[str, Any] = {}


def _lazy_processor():  # type: ignore[no-untyped-def]
    if "processor" not in _lazy_cache:
        raise RuntimeError("processor is not initialised")
    return _lazy_cache["processor"]


def register_processor(processor) -> None:  # type: ignore[no-untyped-def]
    _lazy_cache["processor"] = processor
