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
    symbol = normalize_symbol(raw_token)
    if not symbol:
        return
    duration = (end_index - start_index) * frame_duration
    if symbol not in {"SIL", "SP", "NSN"} and duration < min_duration:
        return
    start = round(start_index * frame_duration, 3)
    end = round(end_index * frame_duration, 3)
    confidence = None
    segment = probabilities[start_index:end_index, token_id]
    if segment.numel() > 0:
        confidence = float(torch.mean(segment).item())
        if confidence < conf_threshold and symbol not in {"SIL", "SP", "NSN"}:
            return
    spans.append((symbol, start, end, confidence))


_lazy_cache: dict[str, Any] = {}


def _lazy_processor():  # type: ignore[no-untyped-def]
    if "processor" not in _lazy_cache:
        raise RuntimeError("processor is not initialised")
    return _lazy_cache["processor"]


def register_processor(processor) -> None:  # type: ignore[no-untyped-def]
    _lazy_cache["processor"] = processor
