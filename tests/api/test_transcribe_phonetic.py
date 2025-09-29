"""API behaviour tests with a stubbed phoneme pipeline."""
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.main import app
from phoneme.pipeline import KanaToken, Phoneme, PhonemeResult

client = TestClient(app)


@contextmanager
def stub_pipeline(return_value: PhonemeResult, captured: list[dict] | None = None) -> Iterator[None]:
    import phoneme.pipeline

    async def _stub(self, audio_bytes: bytes) -> PhonemeResult:
        if captured is not None:
            captured.append({"audio_size": len(audio_bytes)})
        return return_value

    original = phoneme.pipeline.PhonemePipeline.transcribe
    phoneme.pipeline.PhonemePipeline.transcribe = _stub  # type: ignore[assignment]
    try:
        yield
    finally:
        phoneme.pipeline.PhonemePipeline.transcribe = original  # type: ignore[assignment]


def _fake_wave_bytes(duration_seconds: float = 0.5) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    sample_rate = 16_000
    n_frames = int(sample_rate * duration_seconds)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
    return buffer.getvalue()


def test_transcribe_phonetic_returns_stubbed_response() -> None:
    result = PhonemeResult(
        phones=[
            Phoneme(symbol="SIL", start=0.0, end=0.1, confidence=None),
            Phoneme(symbol="M", start=0.1, end=0.2, confidence=0.98),
        ],
        kana=[
            KanaToken(value="", start=0.0, end=0.1),
            KanaToken(value="ム", start=0.1, end=0.2),
        ],
    )

    with stub_pipeline(result):
        response = client.post(
            "/transcribe_phonetic",
            files={"audio": ("sample.wav", _fake_wave_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kana_text"] == "ム"
    assert payload["phones"][1]["p"] == "M"
    assert payload["kana"][1]["k"] == "ム"


def test_option_flags_adjust_kana_rendering() -> None:
    result = PhonemeResult(
        phones=[
            Phoneme(symbol="TH", start=0.0, end=0.1, confidence=0.75),
            Phoneme(symbol="EY", start=0.1, end=0.2, confidence=0.8),
            Phoneme(symbol="T", start=0.2, end=0.3, confidence=0.85),
        ],
    )
    captured: list[dict] = []

    with stub_pipeline(result, captured):
        response = client.post(
            "/transcribe_phonetic",
            files={"audio": ("sample.wav", _fake_wave_bytes(), "audio/wav")},
            data={"style": "loan", "final_c_t": "tsu", "th": "zu"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kana_text"] == "ズメイツ"
    assert [token["k"] for token in payload["kana"]] == ["ズ", "メイ", "ツ"]
    assert captured and captured[0]["audio_size"] > 0
