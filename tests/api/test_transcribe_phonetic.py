"""/transcribe_phonetic エンドポイントのテスト。"""
from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_pipeline, new_request_id
from api.schemas import PHONEME_MAX_BYTES
from asr.pipeline import PhonemeSpan, ShortAudioError, TranscriptionMetrics

client = TestClient(app)


class StubPipeline:
    """テスト用の固定レスポンスを返すパイプライン。"""

    async def transcribe(self, audio_bytes: bytes, *, req_id: str):  # type: ignore[override]
        phones = [
            PhonemeSpan(symbol="n", start=0.0, end=0.2, confidence=0.9),
            PhonemeSpan(symbol="iː", start=0.2, end=0.4, confidence=0.8),
            PhonemeSpan(symbol="d", start=0.4, end=0.6, confidence=0.85),
        ]
        metrics = TranscriptionMetrics(
            req_id=req_id,
            model_id="stub",
            revision="stub",
            device="cpu",
            cold_start_ms=None,
            inference_ms=12.3,
            total_ms=18.7,
            chunk_ms=320,
            overlap=0.5,
            conf_threshold=0.3,
            min_phone_ms=40,
            min_input_ms=320,
            reject_ms=0,
        )
        return phones, metrics


class ShortAudioPipeline:
    async def transcribe(self, audio_bytes: bytes, *, req_id: str):  # type: ignore[override]
        raise ShortAudioError(50.0, 120.0)


def _fake_wav(duration: float = 0.5) -> bytes:
    buffer = io.BytesIO()
    sample_rate = 16_000
    total_frames = int(sample_rate * duration)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * total_frames)
    return buffer.getvalue()


def test_transcribe_phonetic_returns_kana_sequence(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, get_pipeline, StubPipeline)
    monkeypatch.setitem(app.dependency_overrides, new_request_id, lambda: "test")

    try:
        response = client.post(
            "/transcribe_phonetic",
            files={"audio": ("sample.wav", _fake_wav(), "audio/wav")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["kana_text"] == "ニード"
        assert payload["kana_text_readable"] == "ニード"
        assert payload["kana_text_strict"] == "ニード"
        assert payload["kana_ops"]
        assert "type" in payload["kana_ops"][0]
        assert [item["p"] for item in payload["phones"]] == ["n", "iː", "d"]
        assert payload["params"]["conf_threshold"] == 0.3
        assert payload["params"]["min_phone_ms"] == 40
        assert payload["params"]["long_vowel_ms"] == 140
        assert payload["params"]["min_input_ms"] == 320
        assert payload["params"]["reject_ms"] == 0
    finally:
        app.dependency_overrides.pop(get_pipeline, None)
        app.dependency_overrides.pop(new_request_id, None)


def test_transcribe_phonetic_accepts_option_fields(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, get_pipeline, StubPipeline)
    try:
        response = client.post(
            "/transcribe_phonetic",
            files={"audio": ("sample.wav", _fake_wav(), "audio/wav")},
            data={
                "style": "natural",
                "long_vowel_level": "0",
                "sokuon_level": "0",
                "r_coloring": "1",
                "final_c_t": "tsu",
                "th": "zu",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["phones"]
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


def test_transcribe_phonetic_rejects_large_payload(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, get_pipeline, StubPipeline)
    big_payload = b"0" * (PHONEME_MAX_BYTES + 1)
    response = client.post(
        "/transcribe_phonetic",
        files={"audio": ("large.wav", big_payload, "audio/wav")},
    )
    assert response.status_code == 413


def test_transcribe_phonetic_rejects_short_audio(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, get_pipeline, ShortAudioPipeline)
    monkeypatch.setitem(app.dependency_overrides, new_request_id, lambda: "short")

    try:
        response = client.post(
            "/transcribe_phonetic",
            files={"audio": ("sample.wav", _fake_wav(0.05), "audio/wav")},
        )

        assert response.status_code == 400
        payload = response.json()
        assert payload["detail"]["error"] == "short_audio"
    finally:
        app.dependency_overrides.pop(get_pipeline, None)
        app.dependency_overrides.pop(new_request_id, None)


def test_transcribe_phonetic_rejects_invalid_mime() -> None:
    response = client.post(
        "/transcribe_phonetic",
        files={"audio": ("sample.bin", b"abc", "application/octet-stream")},
    )
    assert response.status_code == 415
