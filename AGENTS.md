# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts production code: `api/` (FastAPI routes), `asr/` (wav2vec2 pipeline), `kana/` (phoneme-to-kana rules), plus shared helpers under `core/` and `common/`.
- `backend/` and `client/` provide deployment scaffolding; `assets/` stores bundled language models; `audio_samples/` offers curated WAV fixtures for smoke tests.
- Tests reside in `tests/`, mirroring the source layout (`tests/api`, `tests/asr`, `tests/test_kana_mapping.py`). Batch evaluation results land in `results/`, with `latest` symlinked to the most recent run.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create/enter the development environment.
- `pip install -r requirements.txt` — install runtime and tooling dependencies (torch, transformers, FastAPI, pytest).
- `make run` — start the API locally via uvicorn on `http://localhost:8000` with autoreload.
- `pytest` or `pytest tests/api/test_transcribe_phonetic.py` — run the full suite or target API regressions.
- `./run_eval.sh` — execute the batch regression harness across `audio_samples/`; artifacts are written to `results/<timestamp>/`.

## Coding Style & Naming Conventions
- Python 3.11 codebase; Black and Ruff enforce 100-character lines, double quotes, and modern lint rules. Run `ruff check .` / `ruff format .` before submitting.
- Use descriptive snake_case for modules, functions, and variables; reserve PascalCase for dataclasses and Pydantic schemas.
- Keep FastAPI schemas in `src/api/schemas.py` and app-level DTOs in `src/app/schemas.py`; align new modules with this separation.

## Testing Guidelines
- Pytest is configured via `pyproject.toml` (asyncio mode auto, `src` added to `PYTHONPATH`).
- Name new tests `test_<feature>.py` and collocate with the corresponding module tree.
- For phoneme/kana regressions, add WAV fixtures to `audio_samples/` and cover them with API-level assertions or targeted unit tests.
- Run `pytest` prior to PRs; if batch evals are skipped, call that out in the PR description.

## Commit & Pull Request Guidelines
- Follow the Conventional Commit style seen in history (`feat:`, `fix(asr):`, `chore:`) with concise, present-tense summaries.
- Reference issues or tickets in the body, and list critical test commands executed.
- PRs should outline purpose, affected components (`api`, `asr`, `kana`, etc.), and include screenshots or JSON snippets for user-visible changes.
- Avoid force pushes after review starts; prefer incremental commits addressing feedback for reviewer clarity.
