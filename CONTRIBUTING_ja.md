# verl への貢献

verl への貢献をご検討いただき、ありがとうございます！バグ修正、機能強化、ドキュメントの改善、フィードバックなど、あらゆる種類の貢献を歓迎します。経験豊富な開発者でも、初めてのオープンソースプロジェクトでも、あなたの助けは貴重です。

あなたのサポートは多くの形を取ることができます：
- 問題や予期しない動作を報告する
- 新機能を提案または実装する
- ドキュメントを改善または拡張する
- プルリクエストをレビューし、他の貢献者を支援する
- 宣伝する：ブログ投稿、ソーシャルメディアで verl を共有したり、リポジトリに ⭐ を付ける

## 貢献する問題を見つける

参加方法をお探しですか？以下の問題をチェックしてください：
- [Good first issues](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [Call for contribution](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22call%20for%20contribution%22)

さらに、[RFC](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3ARFC) と [Roadmap](https://github.com/volcengine/verl/issues?q=state%3Aopen%20label%3A%22roadmap%22) を通じて開発計画とロードマップを学ぶことができます。

## 開発

- **Python のみ**: `pip install -e .[test,vllm]` または `pip install -e .[test,sglang]` で verl をインストールし、迅速に反復開発。完全な依存関係セットアップについては、verl [インストールドキュメント](https://verl.readthedocs.io/en/latest/start/install.html)をチェックしてください。

## コードリンティングとフォーマット

コードの一貫性を保つために pre-commit を使用しています。セットアップ方法：

```bash
pip install pre-commit
pre-commit install
# ステージされた変更に対して
pre-commit run
# リポジトリ内のすべてのファイルに対して
pre-commit run --all-files
# pre-commit で特定のフックを実行
# pre-commit run --all-files --show-diff-on-failure --color=always <hood-id>
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

## テスト

テストスイートは GitHub Actions で実行されます。詳細については、以下のワークフローをチェックしてください：
- [GPU ユニットテスト](https://github.com/volcengine/verl/blob/main/.github/workflows/gpu_unit_tests.yml)
- [CPU ユニットテスト](https://github.com/volcengine/verl/blob/main/.github/workflows/cpu_unit_tests.yml)
- [vLLM テスト](https://github.com/volcengine/verl/blob/main/.github/workflows/vllm.yml)
- [SGLang テスト](https://github.com/volcengine/verl/blob/main/.github/workflows/sgl.yml)

### CI テストの追加

可能であれば、新機能に対して CI テストを追加してください：

1. 最も関連性の高いワークフロー yml ファイルを見つけます。これは通常、`hydra` デフォルト設定（例：`ppo_trainer`、`ppo_megatron_trainer`、`sft_trainer` など）に対応します。
2. まだ含まれていない場合は、関連するパスパターンを `paths` セクションに追加します。
3. テストスクリプトのワークロードを最小化します（例については既存のスクリプトを参照）。

## ドキュメントのビルド

```bash
# verl が PYTHONPATH にあることを確認します。例：
pip install -e .[test]

# ドキュメント依存関係をインストール
pip install -r requirements-docs.txt

# HTML ドキュメントを生成
make clean
make html

# ローカルでプレビュー
python -m http.server -d _build/html/
```

ブラウザで http://localhost:8000 を開いてドキュメントを確認してください。

## プルリクエストとコードレビュー

PR を提出していただき、ありがとうございます！レビューを効率化するために：
- タイトル形式とチェックリストについては、プルリクエストテンプレートに従ってください
- pre-commit リントルールに従い、すべてのチェックが通ることを確認してください
- ユーザー向けの変更についてはドキュメントを更新してください
- CI ワークフローでテストを追加または更新するか、テストが適用されない理由を説明してください

## ライセンス

詳細については [LICENSE](https://github.com/volcengine/verl/blob/main/LICENSE) ファイルをご覧ください。

## ありがとうございます

verl への貢献に感謝します。あなたの努力により、プロジェクトがより強力でユーザーフレンドリーになります。ハッピーコーディング！
