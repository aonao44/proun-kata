簡略版 README.md（音素分析ツール削除）
bashcat > README.md << 'EOF'

# proun-kata - 英語音声 → カタカナ変換 API

英語音声を音素認識してカタカナに変換する API システム

## 🎯 機能

- **Wav2Vec2 による高精度音素認識**
- **音素 → カタカナ自動変換**
- **リエゾン・音の繋がりに対応**
- **バッチテスト機能**（43 サンプル自動実行）
- **JSON 形式での詳細結果出力**

---

## 📋 必要要件

- Python 3.8 以上
- ffmpeg
- jq（JSON 処理用）
- 8GB 以上の RAM 推奨
- インターネット接続（初回モデルダウンロード時）

---

## 🚀 セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/YOUR_USERNAME/proun-kata.git
cd proun-kata
2. 仮想環境の作成
bashpython -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
3. 依存関係のインストール
bashpip install -r requirements.txt
requirements.txtの内容:
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0
jinja2>=3.1.0
4. ffmpegのインストール
bash# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
5. jqのインストール（テストスクリプト用）
bash# macOS
brew install jq

# Ubuntu/Debian
sudo apt install jq

# Windows
choco install jq

▶️ 基本的な使い方
APIサーバーの起動
bash# 仮想環境を有効化
source .venv/bin/activate

# パラメータを設定
export CONF_THRESHOLD=0.05
export MIN_PHONE_MS=20
export REJECT_MS=0

# サーバー起動
python -m uvicorn --app-dir src api.app:app --host 0.0.0.0 --port 8001
起動確認:
bashcurl http://localhost:8001/
# → {"message":"Pronunciation API is running"}
単一ファイルのテスト
bash# 音声ファイルを送信
curl -F "audio=@your_audio.wav;type=audio/wav" \
  http://localhost:8001/transcribe_phonetic | jq .
```
