# Configuration Guide

Set environment variables before launching the FastAPI app. Copy `.env.example` to `.env` and
customise for the local workstation.

- `PHONEME_ENGINE`: Choose the recognition backend (`julius` or `wav2vec2`).
- `PHONEME_ENGINE_URL`: Host:port pair for the external recogniser when running Julius in module mode.
- `VAD_ENABLED`: `on` enables optional VAD trimming prior to phoneme decoding.

Future iterations may introduce additional knobs for kana rulesets and streaming mode.
