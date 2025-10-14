# Repository Guidelines

## Project Structure & Module Organization
- `src/`: 音素認識からカタカナ変換までの実装を kata ごとのフォルダーに配置し、例として `src/stress_rules/` が対応するロジックを保持します。
- `src/common/`: 複数 kata で再利用する補助ユーティリティをまとめ、各 kata 間の依存を避けてください。
- `tests/`: `tests/test_<kata>.py` で対応する `src/<kata>/` を検証します。フィクスチャは `tests/fixtures/` に置きます。
- `assets/`: IPA テーブルや音声サンプルなどの補助データを管理し、追加時は `assets/README.md` に出典を追記します。
- `backend/` と `client/`: FastAPI エンドポイントとクライアント試作 UI を格納しています。API 仕様変更時は両方を同時に更新してください。

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: 仮想環境を初期化して依存関係の汚染を防ぎます。
- `python -m pip install -r requirements.txt`: サーバー側の依存を同期します。リスト変更後は必ず実行してください。
- `pytest`: 全テストを実行します。特定 kata のみ検証する場合は `pytest tests/test_stress_rules.py -k edge_case` のように絞り込みます。
- `ruff check src tests` / `ruff format src tests`: コードスタイルの検証と整形を行い、CI と同一の設定を再現します。
- `make run`: `uvicorn` で FastAPI をホットリロード起動し、エンドツーエンドの動作確認に利用します。

## Coding Style & Naming Conventions
- Python は PEP 8 に準拠し、インデントは 4 スペース、行長は 100 文字以内を維持します。
- モジュール・パッケージはスネークケース、クラスはパスカルケース、テスト関数は挙動を説明する英語フレーズで命名します。
- 公開関数には型ヒントを付与し、副作用を避けた純粋関数として実装することを推奨します。
- 変換ルールやマッピングの定数は `src/<kata>/constants.py` などに整理し、コメントで参照規格を明記します。

## Testing Guidelines
- テストは `pytest` の `@pytest.mark.parametrize` を活用してストレスバリエーションを網羅し、回帰を防ぎます。
- 新規ロジックを追加する際は先に失敗するテストを作成し、想定発音（例: `make it`, `can I` など）で検証します。
- 行カバレッジは 90% 以上を維持し、閾値を変更する場合は `coverage.xml` を更新して理由を共有してください。
- CI での一貫性を確保するため、`pytest --maxfail=1 --disable-warnings` でのローカル実行も推奨します。

## Commit & Pull Request Guidelines
- Conventional Commits (`feat:`, `fix:`, `docs:` など) を採用し、1 コミットにつき 1 つの kata もしくは修正に集中させます。
- プルリクエストには対象 kata、主要な挙動変更、関連 Issue、追加コマンドを記載し、動作確認結果を添えます。
- 振る舞いが変わる場合はサンプル音声に対する before/after カタカナを示し、レビュアーの検証負荷を下げます。
- CI がグリーンになるまでマージを保留し、レビュー依頼後は指摘事項に対するフォローアップを明示します。

## Security & Configuration Tips
- 機密情報は `.env.local` に保存し、`.env.example` に安全なデフォルトを提示します。新しい環境変数は `docs/configuration.md` に追記してください。
- 音声データや推論結果を永続化しないのが既定方針です。ログにはリクエスト ID、モデル ID、ルールセット ID のみを残します。
- 外部モデルや辞書を追加する場合はライセンスを確認し、`docs/architecture.md` にアーキテクチャ差分を整理してください。
