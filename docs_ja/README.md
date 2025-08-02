# verl 日本語ドキュメント

verl（Volcano Engine Reinforcement Learning for LLMs）の包括的な日本語ドキュメントへようこそ。

## 概要

verl は大規模言語モデル（LLM）のポストトレーニング用に設計された、柔軟で効率的な本格運用対応の強化学習（RL）トレーニングフレームワークです。[HybridFlow](https://arxiv.org/abs/2409.19256v2) 論文のオープンソース実装版です。

## ドキュメント構成

### 基本ガイド
- [インストールガイド](start/install.md) - verl のインストール方法
- [クイックスタートガイド](start/quickstart.md) - GSM8K データセットでの PPO トレーニング
- [マルチノード設定](start/multinode.md) - 複数ノードでの実行方法

### プログラミングガイド
- [HybridFlow プログラミングガイド](hybrid_flow.md) - verl の核となるプログラミングモデル
- [シングルコントローラー](single_controller.md) - シングルコントローラーアーキテクチャ

### アルゴリズム
- [PPO](algo/ppo.md) - Proximal Policy Optimization
- [GRPO](algo/grpo.md) - Group Relative Policy Optimization
- [DAPO](algo/dapo.md) - Direct Alignment from Preference Optimization
- [その他のアルゴリズム](algo/) - SPIN、SPPO、エントロピー機構など

### パフォーマンス最適化
- [パフォーマンスチューニングガイド](web_content/performance_tuning_guide.md) - 包括的な性能最適化ガイド
- [デバイス調整](perf/device_tuning.md) - GPU とハードウェア最適化
- [大規模モデル対応](perf/dpsk.md) - DeepSeek 671B などの大規模モデル

### ワーカーとバックエンド
- [FSDP ワーカー](workers/fsdp_workers.md) - PyTorch FSDP バックエンド
- [Megatron-LM ワーカー](workers/megatron_workers.md) - Megatron-LM バックエンド
- [SGLang ワーカー](workers/sglang_worker.md) - SGLang 推論バックエンド

### 高度な機能
- [チェックポイント](advance/checkpoint.md) - モデルの保存と復元
- [LoRA サポート](advance/ppo_lora.md) - Low-Rank Adaptation
- [マルチターン対話](sglang_multiturn/multiturn.md) - 複数ターンの対話システム

### Web コンテンツ統合
- [パフォーマンスチューニングガイド](web_content/performance_tuning_guide.md) - 公式サイトから統合された詳細ガイド
- [インストールガイド詳細版](web_content/installation_guide.md) - 拡張インストール手順
- [クイックスタートガイド詳細版](web_content/quickstart_guide.md) - 詳細なクイックスタート

## 主な特徴

### 柔軟性と使いやすさ
- **多様な RL アルゴリズムの簡単な拡張**: ハイブリッドプログラミングモデルにより、複雑なポストトレーニングデータフローの柔軟な表現と効率的な実行が可能
- **既存 LLM インフラとのシームレスな統合**: 計算とデータの依存関係を分離し、FSDP、Megatron-LM、vLLM、SGLang などの既存フレームワークとの統合を実現
- **柔軟なデバイスマッピング**: 効率的なリソース利用と異なるクラスターサイズでのスケーラビリティのため、モデルを異なる GPU セットに配置可能

### 高性能
- **最先端のスループット**: SOTA LLM トレーニングと推論エンジンの統合により高いスループットを実現
- **3D-HybridEngine による効率的なアクターモデル再分散**: メモリ冗長性を排除し、トレーニングと生成フェーズ間の遷移時の通信オーバーヘッドを大幅削減

## サポートされる構成

### トレーニング
- **FSDP** および **Megatron-LM** (オプション)

### 推論
- **vLLM**、**SGLang**、**TGI**

### モデル
- Qwen-3、Qwen-2.5、Llama3.1、Gemma2、DeepSeek-LLM など HuggingFace Transformers 互換モデル

## 貢献

verl はオープンソースプロジェクトです。貢献を歓迎します！

- [貢献ガイド](../CONTRIBUTING.md)
- [GitHub リポジトリ](https://github.com/volcengine/verl)
- [Slack コミュニティ](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)

## ライセンス

Apache License 2.0 の下でライセンスされています。詳細は [LICENSE](../LICENSE) ファイルをご覧ください。
