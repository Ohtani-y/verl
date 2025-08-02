<div align="center">
 👋 皆さん、こんにちは！ 
    verl は <b>ByteDance Seed チーム</b>によって開始され、verl コミュニティによって維持されている RL トレーニングライブラリです。
    <br>
    <br>
</div>

<div align="center">

<a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLMs</h1>

verl は大規模言語モデル（LLM）のための柔軟で効率的、本格運用対応の RL トレーニングライブラリです。

verl は **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** 論文のオープンソース版です。

verl は柔軟で使いやすく、以下の特徴があります：

- **多様な RL アルゴリズムの簡単な拡張**: ハイブリッドコントローラープログラミングモデルにより、複雑なポストトレーニングデータフローの柔軟な表現と効率的な実行が可能。GRPO、PPO などの RL データフローを数行のコードで構築できます。

- **既存 LLM インフラとのモジュラー API によるシームレスな統合**: 計算とデータの依存関係を分離し、FSDP、Megatron-LM、vLLM、SGLang などの既存 LLM フレームワークとのシームレスな統合を実現

- **柔軟なデバイスマッピング**: 効率的なリソース利用と異なるクラスターサイズでのスケーラビリティのため、モデルを異なる GPU セットに配置することをサポート

- 人気の HuggingFace モデルとの即座の統合

verl は高速で、以下の特徴があります：

- **最先端のスループット**: SOTA LLM トレーニングと推論エンジンの統合、および SOTA RL スループット

- **3D-HybridEngine による効率的なアクターモデル再分散**: メモリ冗長性を排除し、トレーニングと生成フェーズ間の遷移時の通信オーバーヘッドを大幅に削減

</p>

## ニュース
- [2025/07] [ReTool](https://arxiv.org/pdf/2504.11536) レシピが完全にオープンソース化されました。[ブログ](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
- [2025/07] 初回 verl ミートアップが 7月16日に ICML Vancouver で開催されます！ICML にご参加の方は[ぜひお越しください](https://lu.ma/0ek2nyao)！（現地参加のみ）
- [2025/07] 7/8 の [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) での verl 基調講演、7/11 の LF AI & Data Singapore による [Agent for SWE meetup](https://lu.ma/e498qhsi) での verl & verl-agent プロジェクトアップデート
- [2025/06] Megatron バックエンドを使用した verl により、[DeepSeek-671b や Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html) などの大規模 MoE モデルが利用可能になりました
- [2025/06] verl チームが 6月7日の [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) で最新のプロジェクトアップデートを提供します。北京で開発チームにお会いしましょう！
- [2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) 技術レポートがリリースされました！verl でトレーニングされた Seed-Thinking-v1.5 は、AIME 2024 で 86.7、Codeforces で 55.0、GPQA で 77.3 を達成し、STEM とコーディングにおける優れた推論能力を実証しています。推論タスクを超えて、この手法は多様な領域での顕著な汎化を示しています。
- [2025/03] [DAPO](https://dapo-sia.github.io/) は、Qwen2.5-32B 事前トレーニングモデルに基づいて AIME 2024 で 50 ポイントを達成し、DeepSeek の GRPO（DeepSeek-R1-Zero-Qwen-32B）による以前の SOTA を上回るオープンソース SOTA RL アルゴリズムです。DAPO のトレーニングは完全に verl によって支えられており、再現コードは現在 `recipe/dapo` で利用可能です。
<details><summary> more... </summary>
<ul>
  <li> [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.</li>
  <li>[2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl! PF-PPO enhances policy learning efficiency and robustness by filtering potentially noisy reward signals and reusing high-quality experiences via a replay buffer.</li>
  <li>[2025/04] We will give a tutorial about latest post-training techniques and programming guide for verl at [ICLR 2025 Expo](https://iclr.cc/virtual/2025/calendar?filter_events=Expo+Talk+Panel&filter_rooms=), [SCI-FM workshop](https://open-foundation-model.github.io/) and [LMSys afterparty](https://lu.ma/d23nyynm). Talk materials available [here](https://github.com/eric-haibin-lin/verl-community/tree/main/iclr25). </li>
  <li>[2025/03] verl v0.3.0.post1 is released! See [release note](https://github.com/volcengine/verl/releases/) for details. It achieves [~1.4x speedup](https://tongyx361.github.io/blogs/posts/verl-intro/#/verl-flexible-and-efficient-rl-for-llms) compared to prev versions.</li>
  <li>[2025/05] verl will be presented at [A2M Shanghai](https://a2m.msup.com.cn/home/?aid=4488&city=shanghai) on 5/16 - 5/17.</li>
  <li>[2025/05] verl will be presented at [GOSIM x PyTorch Day 2025](https://paris2025.gosim.org/). See you in Paris! </li>
  <li>[2025/03] We introduced the programming model of verl at the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg) and [verl intro and updates](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl-lmsys-meetup.pdf) at the [SGLang-LMSYS Org Meetup](https://lu.ma/ntjrr7ig) in Sunnyvale mid-March.</li>
  <li>[2025/03] We will present verl(HybridFlow) at EuroSys 2025. See you in Rotterdam!</li>
  <li>[2025/02] verl v0.2.0.post2 is released!</li>
  <li>[2025/02] We presented verl in the <a href="https://lu.ma/ji7atxux">Bytedance/NVIDIA/Anyscale Ray Meetup</a>. See you in San Jose!</li>
  <li>[2025/01] [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) is released with SOTA-level performance on LLM & VLM. The RL scaling preview model is trained using verl, reaching OpenAI O1-level performance on math benchmarks (70.0 pass@1 on AIME).</li>
  <li>[2024/12] verl is presented at Ray Forward 2024. Slides available <a href="https://github.com/eric-haibin-lin/verl-community/blob/main/slides/Ray_Forward_2024_%E5%B7%AB%E9%94%A1%E6%96%8C.pdf">here</a></li>
  <li>[2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024. <a href="https://github.com/eric-haibin-lin/verl-data/tree/neurips">Slides</a> and <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">video</a> available.</li>
  <li>[2024/10] verl is presented at Ray Summit. <a href="https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37">Youtube video</a> available.</li>
  <li>[2024/08] HybridFlow (verl) is accepted to EuroSys 2025.</li>
</ul>   
</details>

## 主な機能

- トレーニング用の **FSDP**、**FSDP2**、**Megatron-LM**
- ロールアウト生成用の **vLLM**、**SGLang**、**HF Transformers**
- Hugging Face Transformers と Modelscope Hub との互換性：[Qwen-3](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh)、Qwen-2.5、Llama3.1、Gemma2、DeepSeek-LLM など
- 教師ありファインチューニング
- [PPO](examples/ppo_trainer/)、[GRPO](examples/grpo_trainer/)、[ReMax](examples/remax_trainer/)、[REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm)、[RLOO](examples/rloo_trainer/)、[PRIME](recipe/prime/)、[DAPO](recipe/dapo/)、[DrGRPO](recipe/drgrpo)、[KL_Cov & Clip_Cov](recipe/entropy) などによる強化学習
  - 数学、[コーディング](https://github.com/volcengine/verl/tree/main/recipe/dapo) などのモデルベース報酬と関数ベース報酬（検証可能な報酬）をサポート
  - Qwen2.5-vl、Kimi-VL による視覚言語モデル（VLM）と[マルチモーダル RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh) をサポート
  - [ツール呼び出しを伴うマルチターン](https://github.com/volcengine/verl/tree/main/examples/sglang_multiturn)
- [自己対戦選好最適化（SPPO）](https://github.com/volcengine/verl/tree/main/recipe/sppo) などの LLM アライメントレシピ
- Flash attention 2、[シーケンスパッキング](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh)、DeepSpeed Ulysses による[シーケンス並列化](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh)、[LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh)、[Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh) のサポート
- [エキスパート並列化](https://github.com/volcengine/verl/pull/1467)により 671B モデルと数百の GPU までスケール
- メモリ節約のためのマルチ GPU [LoRA RL](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) サポート
- wandb、swanlab、mlflow、tensorboard による実験追跡

## Upcoming Features and Changes

- Q3 Roadmap https://github.com/volcengine/verl/issues/2388
- DeepSeek 671b optimizations with Megatron https://github.com/volcengine/verl/issues/1033
- Multi-turn rollout and tools using optimizations https://github.com/volcengine/verl/issues/1882
- [Agent integration](https://github.com/volcengine/verl/tree/main/verl/experimental/agent_loop)
- Async and off-policy architecture https://github.com/volcengine/verl/pull/2231
- List of breaking changes since v0.4 https://github.com/volcengine/verl/discussions/2270

## はじめに

<a href="https://verl.readthedocs.io/en/latest/index.html"><b>ドキュメント</b></a> | <a href="docs_ja/README.md"><b>日本語ドキュメント</b></a>

**クイックスタート:**

- [インストール](https://verl.readthedocs.io/en/latest/start/install.html) | [日本語版](docs_ja/start/install.md)
- [クイックスタート](https://verl.readthedocs.io/en/latest/start/quickstart.html) | [日本語版](docs_ja/start/quickstart.md)
- [プログラミングガイド](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [技術講演](https://hcqnc.xetlk.com/sl/3vACOK)（中国語）| [日本語版](docs_ja/hybrid_flow.md)
- [verl での PPO](https://verl.readthedocs.io/en/latest/algo/ppo.html) | [日本語版](docs_ja/algo/ppo.md)
- [verl での GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html) | [日本語版](docs_ja/algo/grpo.md)

**PPO サンプルのステップバイステップ実行:**

- [ポストトレーニング用データの準備](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) | [日本語版](docs_ja/preparation/prepare_data.md)
- [データセット用報酬関数の実装](https://verl.readthedocs.io/en/latest/preparation/reward_function.html) | [日本語版](docs_ja/preparation/reward_function.md)
- [PPO サンプルアーキテクチャ](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html) | [日本語版](docs_ja/examples/ppo_code_architecture.md)
- [設定の説明](https://verl.readthedocs.io/en/latest/examples/config.html) | [日本語版](docs_ja/examples/config.md)

**再現可能なアルゴリズムベースライン:**

- [コーディング、数学での RL パフォーマンス](https://verl.readthedocs.io/en/latest/algo/baseline.html) | [日本語版](docs_ja/algo/baseline.md)

**For code explanation and advance usage (extension):**

- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)

- Advanced Usage and Extension
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
  - [Search Tool Integration](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
  - [Sandbox Fusion Integration](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)

**Blogs from the community**

- [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
- [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
- [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
- [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
- [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
- [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
- [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
- [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

## パフォーマンスチューニングガイド

オンポリシー RL アルゴリズムにとってパフォーマンスは不可欠です。パフォーマンス最適化を支援するため、詳細な[パフォーマンスチューニングガイド](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)（[日本語版](docs_ja/web_content/performance_tuning_guide.md)）を作成しました。

## Upgrade to vLLM >= v0.8.2

verl now supports vLLM>=0.8.2 when using FSDP as the training backend. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for the installation guide and more information. Please avoid vllm 0.7.x, which contains bugs that may lead to OOMs and unexpected errors.

## Use Latest SGLang

SGLang is fully supported with verl, and SGLang RL Group is working extensively on building unique features, including multi-turn agentic RL, VLM RLHF, server-based RL, and partial rollout. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.

## Upgrade to FSDP2

verl is fully embracing FSDP2! FSDP2 is recommended by torch distributed team, providing better throughput and memory usage, and is composible with other features (e.g. torch.compile). To enable FSDP2, simply use verl main and set the following options:
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```
Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with `actor_rollout_ref.actor.fsdp_config.offload_policy=True`. For more details, see https://github.com/volcengine/verl/pull/1026

## AMD Support (ROCm Kernel)

verl now supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.


## 引用と謝辞

このプロジェクトが役立つと思われる場合は、以下を引用してください：

- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and contributed by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

## Awesome work using verl

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): [A no human curated data self-play framework for reasoning](https://arxiv.org/abs/2505.03335) ![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): An easy and efficient RL post-training framework for Agentic Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Simple-Efficient/RL-Factory)
- [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs. Code release is in progress...
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent): MemAgent: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent ![GitHub Repo stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent)
- [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS): A Post-training recipe for scaling RL on Advanced Reasoning models ![GitHub Repo stars](https://img.shields.io/github/stars/ChenxinAn-fdu/POLARIS)
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
- [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL)
- [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models ![GitHub Repo stars](https://img.shields.io/github/stars/channel-io/ch-tts-llasa-rl-grpo)
- [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
- [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/gyhdog99/RACRO2)
- [Agent Lightning](https://github.com/microsoft/agent-lightning): A flexible and extensible framework that enables seamless agent optimization for any existing agent framework. ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/agent-lightning)

and many more awesome work listed in [recipe](recipe/README.md).

## 貢献ガイド

[貢献ガイド](CONTRIBUTING.md)（[日本語版](CONTRIBUTING_ja.md)）をご覧ください

## About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society. You can get to know Bytedance Seed better through the following channels👇
<div>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>

</div>
---

We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.
