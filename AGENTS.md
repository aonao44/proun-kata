＃チャットのやり取りは日本語でお願いします

# Repository Guidelines

## Project Structure & Module Organization

- The repository starts lean; place implementation code under `src/` with one folder per kata (e.g. `src/stress_rules/`).
- Mirror the structure in `tests/` so that `tests/test_stress_rules.py` exercises `src/stress_rules/rules.py`.
- Keep supporting assets (IPA tables, sample wordlists, stub audio) inside `assets/` and document provenance in `assets/README.md`.
- Shared utilities belong in `src/common/`; avoid cross-importing between kata folders to keep problems isolated.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate`: bootstrap a virtual environment; run it before executing tools.
- `python -m pip install -r requirements.txt`: sync Python dependencies whenever the list changes.
- `pytest`: run the full suite; scope to a feature with `pytest tests/test_stress_rules.py -k edge_case`.
- `ruff check src tests` and `ruff format src tests`: lint and format code to match the project's ruff configuration.

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation and 100-character lines; let `ruff` and `black` enforce formatting.
- Modules and packages use snake_case (`stress_rules.py`), classes PascalCase, and test functions descriptive (`test_handles_silent_vowels`).
- Keep functions pure where possible; prefer dataclasses for structured pronunciation data and type annotations on every public function.

## Testing Guidelines

- Build tests with `pytest` parametrize to cover stress variants; keep fixtures in `tests/fixtures/`.
- Name files `test_<module>.py` and include docstrings summarizing behaviour.
- Add regression tests before shipping a fix; update `coverage.xml` if thresholds change and keep line coverage above 90%.

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `docs:`) and keep each commit focused on one kata or fix.
- Pull requests must describe the kata tackled, list new commands or configuration files, and link the tracking issue.
- Provide before/after snippets or sample pronunciations when behaviour changes; add notes for additional setup.
- Request review from another contributor and wait for CI green before merging.

## Security & Configuration Tips

- Store secrets (API tokens, cloud endpoints) in `.env.local`; add `.env.local` to `.gitignore`.
- Document new environment variables in `docs/configuration.md` and provide defaults via `.env.example`.

要件定義書（完成版）

1. 目的

英会話教師が生徒の実際の発音を客観的に把握できるよう、発音を補正せずにカタカナ表記で提示する。

2. 基本方針

発音の忠実な再現：音声認識の言語モデル・辞書・単語補正を一切使わない。

音素認識 (phoneme recognition) → カタカナ変換 の直列処理のみ。

禁止事項：正書法正規化、句読点付与、辞書による単語化、Whisper/Vosk 利用。

3. アーキテクチャ
   [Client (Web/RN 録音)]
   └─ WAV(16kHz/mono/16bit)
   │
   ▼
   [Local PC: FastAPI]
   ├─ (任意)VAD で無音除去
   ├─ 音素認識 (Julius or Wav2Vec2 phoneme)
   │ - CTC greedy のみ / LM weight=0
   │ - IPA→ARPAbet 変換・ストレス番号除去
   │ - 無音/雑音ラベルは phones に残すがカタナ空白
   ├─ ルールベース変換 (phones→kana)
   └─ JSON 返却
   │
   [Cloudflare Tunnel / ngrok]
   │
   ▼
   [Client でカタカナ表示]

4. 音素認識エンジン

選択肢 1：Julius 英語 AM、モジュールモード（10500 ポート）で音素列ストリーミング。

選択肢 2：Wav2Vec2-phoneme (CTC), Hugging Face モデル（IPA/ARPAbet）。CTC greedy 必須。

ストリーミング I/F：将来用に /ws/phoneme を予約。MVP は HTTP 一括 POST。

5. 音素前処理

ストレス番号削除（例：IH1 → IH）。

IPA モデル使用時：IPA→ARPAbet マッピング層で統一。

無音/雑音（SIL, NSN, SP）は phones に残し、カタナは空。

6. 音素 → カタカナ変換ルール

CV 合成（子音+母音）。英語にない連結は促音/母音挿入で埋める。

二重母音：EY→「ェイ/メィ」、AY→「ァイ」、OW→「オウ」、OY→「ォイ」、AW→「ァウ」。

語末処理：T→ ッ/ッツ、K→ ック、P→ ップ、S→ ス、N→ ン、R/L→ ル。

特殊ケース：

TH→ ス/ズ（切替可）

flap /ɾ/ → D 扱い

ER→「ァー＋ル」

正規化：連続促音（ッッ）→ ッ 1 つに統合。

7. API 仕様

POST /transcribe_phonetic

入力：audio/wav（multipart/form-data）

オプション：style=rough|loan, final_c_t=xtsu|tsu, th=su|zu

応答例：

{
"phones":[{"p":"M","start":0.12,"end":0.18,"conf":0.82}, ...],
"kana":[{"k":"メ","start":0.12,"end":0.18}, ...],
"kana_text":"メィキッツ"
}

8. クライアント表示

録音 → API 送信 → カタカナ文字列表示。

オプション切替（崩し度/TH/末尾 T）。

簡易タイムライン表示（任意）。

9. 非機能要件

同時ユーザ：2〜5 人。

レイテンシ：短フレーズで 0.3〜0.8 秒。

可用性：PC 依存（Cloudflare/ngrok）。PC 停止で URL 失効。

ログ：処理時間/req ID/model ID/ruleset ID のみ。音声・出力保存なし。

10. テスト計画

素材：make it / can I / would you / want you / tell us / I’m in。

確認：

単語消失なし（phones に残る）。

聴感に近いカタナ。

オプション切替の反映。

同一音声 → 同一出力。

11. アンチ要件

辞書・言語モデルによる補正。

綴り復元、句読点付与。

Whisper/Vosk 利用。

DB 保存、認証。
