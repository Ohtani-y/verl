verl: Volcano Engine Reinforcement Learning for LLMs
========================================================

verl は大規模言語モデル（LLM）のための柔軟で効率的、本格運用対応の RL トレーニングライブラリです。

verl は **HybridFlow: A Flexible and Efficient RLHF Framework** 論文のオープンソース版です。

verl は柔軟で使いやすく、以下の特徴があります：

- **多様な RL アルゴリズムの簡単な拡張**: ハイブリッドコントローラープログラミングモデルにより、複雑なポストトレーニングデータフローの柔軟な表現と効率的な実行が可能。GRPO、PPO などの RL データフローを数行のコードで構築できます。

- **既存 LLM インフラとのモジュラー API によるシームレスな統合**: 計算とデータの依存関係を分離し、FSDP、Megatron-LM、vLLM、SGLang などの既存 LLM フレームワークとのシームレスな統合を実現

- **柔軟なデバイスマッピング**: 効率的なリソース利用と異なるクラスターサイズでのスケーラビリティのため、モデルを異なる GPU セットに配置することをサポート

- 人気の HuggingFace モデルとの即座の統合

verl は高速で、以下の特徴があります：

- **最先端のスループット**: SOTA LLM トレーニングと推論エンジンの統合、および SOTA RL スループット

- **3D-HybridEngine による効率的なアクターモデル再分散**: メモリ冗長性を排除し、トレーニングと生成フェーズ間の遷移時の通信オーバーヘッドを大幅に削減

はじめに
========

.. toctree::
   :maxdepth: 2
   :caption: はじめに

   start/install
   start/quickstart
   start/multinode

プログラミングガイド
==================

.. toctree::
   :maxdepth: 2
   :caption: プログラミングガイド

   hybrid_flow
   single_controller

データ準備
==========

.. toctree::
   :maxdepth: 2
   :caption: データ準備

   preparation/prepare_data
   preparation/reward_function

設定
====

.. toctree::
   :maxdepth: 2
   :caption: 設定

   examples/config
   examples/ppo_code_architecture

アルゴリズム
============

.. toctree::
   :maxdepth: 2
   :caption: アルゴリズム

   algo/ppo
   algo/grpo
   algo/baseline

ワーカー
========

.. toctree::
   :maxdepth: 2
   :caption: ワーカー

   workers/fsdp_workers
   workers/megatron_workers
   workers/sglang_worker

パフォーマンス
==============

.. toctree::
   :maxdepth: 2
   :caption: パフォーマンス

   web_content/performance_tuning_guide
   perf/device_tuning
   perf/dpsk

高度な機能
==========

.. toctree::
   :maxdepth: 2
   :caption: 高度な機能

   advance/checkpoint
   advance/ppo_lora
   sglang_multiturn/multiturn

ハードウェアサポート
==================

.. toctree::
   :maxdepth: 2
   :caption: ハードウェアサポート

   hardware/amd

API リファレンス
===============

.. toctree::
   :maxdepth: 2
   :caption: API リファレンス

   api/index

FAQ
===

.. toctree::
   :maxdepth: 2
   :caption: FAQ

   faq

開発ノート
==========

.. toctree::
   :maxdepth: 2
   :caption: 開発ノート

   dev/index

貢献
====

verl への貢献を歓迎します！

- `GitHub リポジトリ <https://github.com/volcengine/verl>`_
- `Slack コミュニティ <https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA>`_

ライセンス
==========

Apache License 2.0 の下でライセンスされています。

索引と表
========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
