# Configuration Guide

1. コピー: `.env.example` を `.env` に複製します。
2. `PHONEME_BACKEND` は現在 `w2v2` のみサポートしています。
3. `PHONEME_MODEL_ID` / `PHONEME_MODEL_REVISION` / `PHONEME_DEVICE` を環境に合わせて設定します。`PHONEME_MODEL_REVISION` には利用するコミットSHAを入力してください。
4. `CHUNK_MS` と `CHUNK_OVERLAP` は CTC グリーディ推論時のチャンク長とオーバーラップ率です（要件既定値: 320ms / 0.5）。
5. `CONF_THRESHOLD` と `MIN_PHONE_MS` は phone のフィルタリング条件です（要件既定値: 0.30 / 40ms）。
6. `READING_STYLE`（`raw|balanced|natural`）、`LONG_VOWEL_LEVEL`（0–2）、`SOKUON_LEVEL`（0–2）、`R_COLORING`（0/1）を調整すると、“バランス”変換の挙動を変更できます。
7. モデルキャッシュ先を変更したい場合は `HF_HOME` を編集します。
8. Uvicornのバインド先を変更したいときは `UVICORN_HOST` / `UVICORN_PORT` を調整します。

環境設定後は以下でサーバーを起動します。

```
uvicorn --app-dir src api.app:app --reload --port ${UVICORN_PORT:-8000}
```
