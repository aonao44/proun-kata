# Pronunciation Kata Service

FastAPI project that exposes a `/transcribe_phonetic` endpoint for converting raw speech audio into
katakana without lexical substitution. The current build ships with a deterministic stub pipeline so
you can run the API locally while the Julius/wav2vec2 integration is in progress.

## Quickstart

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
2. Copy the sample environment:
   ```bash
   cp .env.example .env
   ```
3. Start the development server:
   ```bash
   make run
   ```
   If you prefer to run uvicorn directly:
   ```bash
   uvicorn --app-dir src app.main:app --reload --port 8000
   # or
   PYTHONPATH=src uvicorn app.main:app --reload --port 8000
   ```

The stub returns a synthetic phoneme/kana sequence based on basic rules until the true recogniser is
integrated.
By default the service runs in stub mode, so no Hugging Face downloads are required. The startup log
prints the active backend (for example `backend=stub`).

## Hugging Face backend (optional)

If you have access to a phoneme CTC model on Hugging Face, point the service at it via the
`PHONEME_MODEL_ID` environment variable. The server falls back to the stub automatically whenever the
model cannot be loaded.

```bash
export PHONEME_MODEL_ID="<your-hf-phoneme-ctc-model>"
hf auth login  # required when the model is private
make run
```

Use `GET http://127.0.0.1:8000/healthz` to confirm which backend is active (`stub` or `hf:<model>`).

## Browser client

With `make run` active, you can try the browser-only client without any build tooling:

1. Install `ffmpeg` so that the backend can transcode browser recordings:
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu / Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```
2. Launch the API with `make run` (or the uvicorn commands shown above).
3. Open `client/web/index.html` directly in your browser (opening via `file://` is fine).
4. 画面には「ファイルを解析」「その場で録音して解析」の 2 タイルが表示されます。音声ファイルをドロップする／録音して **解析する** を押すだけで結果が表示されます。
5. 結果パネルでは `kana_text` が大きく表示され、コピーや Phones/Kana テーブル、折りたたみ式の Raw JSON を確認できます。詳細設定（style / final_c_t / th）は必要なときだけ開いて調整してください。
   <!-- TODO: 新しい UI のスクリーンショットをここに挿入 -->

If you prefer an API-first workflow, open `http://127.0.0.1:8000/docs` and use **POST
/transcribe_phonetic_any**. The endpoint accepts uploads in `webm`, `mp4`, `mp3`, and `wav` formats.

```bash
# Example: send a WebM capture
curl -X POST http://127.0.0.1:8000/transcribe_phonetic_any \
  -F "audio=@sample.webm" \
  -F "style=rough" \
  -F "final_c_t=xtsu" \
  -F "th=su"

# Existing WAV-only endpoint
curl -X POST http://127.0.0.1:8000/transcribe_phonetic \
  -F "audio=@sample.wav" \
  -F "style=rough" \
  -F "final_c_t=xtsu" \
  -F "th=su"
```

## Project Structure

- `src/app/`: FastAPI entrypoint, dependency wiring, and API schemas
- `src/phoneme/`: Pipeline abstractions for phoneme recognition and kana conversion rules
- `src/core/`: Reserved for shared utilities (configuration, logging)
- `tests/`: Pytest suites (stubbed) for API and pipeline behaviour
- `docs/`: Additional documentation (configuration, architecture)
- `assets/`: Future samples, fixtures, or reference tables
