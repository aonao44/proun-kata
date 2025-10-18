# proun-kata - 英語音声 → カタカナ変換 API

英語音声を音素列に変換し、ルールベースでカタカナへ写像する学習教材向けバックエンドです。FastAPI で提供する `/transcribe_phonetic` エンドポイントを中心に、ASR (wav2vec2)、音素→カナ変換、結果の可視化までを一体で提供します。

## システム要件

- **OS**: macOS 12+/Ubuntu 22.04+/Windows 11 (WSL2 推奨)。
- **Python**: 3.11 系（`python --version` で確認）。
- **追加ツール**: `ffmpeg`（音声変換）、`jq`（JSON フィルタ）、`git`。
- **ハードウェア**: 8GB 以上の RAM。GPU は任意（CPU 動作可）。
- **ネットワーク**: 初回起動時に Hugging Face からモデルを取得するため、安定した回線が必要です。

## セットアップ手順

1. **リポジトリの取得**
   ```bash
   git clone https://github.com/YOUR_ORG/proun-kata.git
   cd proun-kata
   ```

2. **Python 仮想環境の準備**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # macOS/Linux
   # .venv\\Scripts\\activate      # Windows(PowerShell)
   ```

3. **Python 依存関係のインストール**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **ネイティブツールの導入**
   - macOS: `brew install ffmpeg jq`
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg jq`
   - Windows: `choco install ffmpeg jq` または WSL 上で apt を利用

5. **初回モデルダウンロード**（任意の WAV を使って一度 API を呼び出すと自動取得されます）。

## 環境変数の調整（任意）

| 変数名 | 既定値 | 説明 |
| --- | --- | --- |
| `CONF_THRESHOLD` | `0.05` | 音素信頼度の下限値 |
| `MIN_PHONE_MS` | `20` | 1 音素の最小長（ミリ秒） |
| `MIN_INPUT_MS` | `320` | 1 チャンクの最小長 |
| `REJECT_MS` | `0` | 音声全体がこの長さ未満だと拒否 |
| `LONG_VOWEL_MS` | `140` | 自動長音化の閾値 |

`.env` は不要ですが、開発用 shell で `export`/`set` してください。

## サーバーの起動

```bash
source .venv/bin/activate
export CONF_THRESHOLD=0.05
export MIN_PHONE_MS=20
python -m uvicorn --app-dir src api.app:app --host 0.0.0.0 --port 8001 --reload
```

- 起動確認: `curl http://localhost:8001/healthz`
- ブラウザ UI: `http://localhost:8001/docs`

## 音声の変換例

```bash
curl -F "audio=@audio_samples/01_liaisons/right_away.wav;type=audio/wav" \
  http://localhost:8001/transcribe_phonetic | jq .
```

レスポンスには `kana_text_readable`（人が読みやすいカタカナ）、`kana_text_strict`（機械判定用）、`kana_ops`（発火した表示ルール）が含まれます。

## テストと検証

- 単体テスト: `pytest`（特定ファイルのみ例: `pytest tests/test_kana_mapping.py`）。
- 型/静的解析: `ruff check .`、自動整形: `ruff format .`。
- 一括評価: `./run_eval.sh` を実行すると `results/<timestamp>/` に JSON と CSV を保存します。HTTP エラーは `HTTP <code>` として記録され、`jq` エラーで停止しません。

## ディレクトリ構成

- `src/api/` — FastAPI ルーター、スキーマ。
- `src/asr/` — wav2vec2 ベースの音素推定、チャンク処理。
- `src/kana/` — 音素→カタカナ変換ロジック、表示ルール。
- `audio_samples/` — 動作確認用 WAV（バッチテストにも利用）。
- `tests/` — Pytest スイート（API・ASR・Kana 単体テスト）。
- `run_eval.sh` — バッチ回帰スクリプト。実行後 `latest/` シンボリックリンクが更新されます。

以上でローカル開発サーバーが起動し、音声ファイルを投入できる状態になります。問題が起きた場合は `logs/`（uvicorn 標準出力）や `results/<timestamp>/` のエラーファイルを参照してください。
