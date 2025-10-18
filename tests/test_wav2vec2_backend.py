"""Regression tests for wav2vec2 backend helpers."""

from __future__ import annotations

import pytest
import torch

from asr.wav2vec2_backend import ShortAudioError, Wav2Vec2Pipeline


def _make_pipeline(**attrs) -> Wav2Vec2Pipeline:
    pipeline = Wav2Vec2Pipeline.__new__(Wav2Vec2Pipeline)
    for key, value in attrs.items():
        setattr(pipeline, key, value)
    return pipeline


def test_prepare_for_model_allows_short_tail_chunk() -> None:
    sample_rate = 16_000
    chunk_ms = 40.0
    samples = int(sample_rate * (chunk_ms / 1000.0))
    chunk = torch.zeros(samples)
    pipeline = _make_pipeline(_reject_ms=120.0, _min_input_ms=60.0)

    padded, original = pipeline._prepare_for_model(chunk, sample_rate)

    assert original == samples
    assert padded.shape[-1] >= original


@pytest.mark.asyncio
async def test_transcribe_rejects_globally_short_audio(monkeypatch) -> None:
    sample_rate = 16_000
    waveform = torch.zeros(int(sample_rate * 0.05))  # 50 ms
    pipeline = _make_pipeline(
        _reject_ms=150.0,
        _min_input_ms=0.0,
        _chunk_ms=20,
        _overlap=0.0,
        _conf_threshold=0.1,
        _min_phone_ms=30.0,
        _device_spec="cpu",
    )

    def fake_load_waveform(_: bytes) -> tuple[torch.Tensor, int]:
        return waveform.clone(), sample_rate

    def fake_ensure_sample_rate(tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        return tensor, sr

    def fail_ready():  # pragma: no cover - should not execute
        raise AssertionError("ensure_ready should not be called for rejected audio")

    monkeypatch.setattr("asr.wav2vec2_backend.load_waveform", fake_load_waveform)
    monkeypatch.setattr("asr.wav2vec2_backend.ensure_sample_rate", fake_ensure_sample_rate)
    monkeypatch.setattr(pipeline, "ensure_ready", fail_ready)

    with pytest.raises(ShortAudioError):
        await pipeline.transcribe(b"dummy", req_id="test")
