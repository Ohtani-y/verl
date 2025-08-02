# 設定ガイド

verl の設定システムと各種パラメータの詳細について説明します。

## 概要

verl は YAML ベースの階層的設定システムを使用しています。設定ファイルは複数のセクションに分かれており、それぞれが異なる機能を制御します。

## 基本構造

### メイン設定ファイル

```yaml
# モデル設定
model:
  path: "models/qwen2.5-7b"
  trust_remote_code: true
  torch_dtype: bfloat16

# データ設定
data:
  train_files: "data/train.jsonl"
  val_files: "data/val.jsonl"
  max_length: 2048
  batch_size: 32

# アルゴリズム設定
algorithm:
  name: ppo
  learning_rate: 1e-6
  epochs: 3

# トレーニング設定
training:
  max_steps: 1000
  save_steps: 100
  output_dir: "outputs/"

# 推論設定
inference:
  backend: vllm
  max_new_tokens: 512
  temperature: 0.7
```

## モデル設定

### 基本パラメータ

```yaml
model:
  # モデルパス（HuggingFace Hub または ローカルパス）
  path: "Qwen/Qwen2.5-7B"
  
  # リモートコードの信頼
  trust_remote_code: true
  
  # データ型
  torch_dtype: bfloat16  # float16, float32, bfloat16
  
  # デバイス配置
  device_map: auto
  
  # 最大シーケンス長
  max_position_embeddings: 32768
```

### 量子化設定

```yaml
model:
  quantization:
    enabled: true
    method: bitsandbytes  # bitsandbytes, gptq, awq
    bits: 4
    double_quant: true
    quant_type: nf4
```

### LoRA 設定

```yaml
model:
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules:
      - q_proj
      - v_proj
      - k_proj
      - o_proj
```

## データ設定

### データソース

```yaml
data:
  # 訓練データ
  train_files:
    - "data/train_part1.jsonl"
    - "data/train_part2.jsonl"
  
  # 検証データ
  val_files: "data/val.jsonl"
  
  # テストデータ
  test_files: "data/test.jsonl"
  
  # データ形式
  data_format: jsonl  # jsonl, json, parquet
```

### データ処理

```yaml
data:
  # 最大長
  max_length: 2048
  
  # バッチサイズ
  batch_size: 32
  
  # シャッフル
  shuffle: true
  
  # ワーカー数
  num_workers: 4
  
  # 前処理
  preprocessing:
    remove_duplicates: true
    filter_by_length: true
    min_length: 10
```

## アルゴリズム設定

### PPO 設定

```yaml
algorithm:
  name: ppo
  
  # 学習率
  learning_rate: 1e-6
  
  # バッチサイズ
  batch_size: 64
  mini_batch_size: 16
  
  # エポック数
  epochs: 3
  
  # PPO パラメータ
  clip_range: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  
  # KL 発散制約
  kl_target: 0.01
  kl_coef: 0.1
  adaptive_kl: true
  
  # グラディエント
  max_grad_norm: 1.0
```

### GRPO 設定

```yaml
algorithm:
  name: grpo
  
  # 基本パラメータ
  learning_rate: 1e-6
  batch_size: 64
  epochs: 3
  
  # GRPO 固有パラメータ
  group_size: 8
  baseline_type: group_mean  # group_mean, group_median
  temperature: 1.0
  
  # 適応的設定
  adaptive_group_size: true
  min_group_size: 4
  max_group_size: 16
```

## トレーニング設定

### 基本設定

```yaml
training:
  # ステップ数
  max_steps: 1000
  
  # 保存頻度
  save_steps: 100
  
  # 評価頻度
  eval_steps: 50
  
  # ログ頻度
  logging_steps: 10
  
  # 出力ディレクトリ
  output_dir: "outputs/experiment_1"
  
  # 再開設定
  resume_from_checkpoint: null
```

### 最適化設定

```yaml
training:
  # オプティマイザ
  optimizer:
    type: adamw
    lr: 1e-6
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1e-8
  
  # スケジューラ
  lr_scheduler:
    type: cosine
    warmup_steps: 100
    total_steps: 1000
  
  # グラディエント
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  gradient_checkpointing: false
```

### メモリ最適化

```yaml
training:
  # 動的バッチサイズ
  use_dynamic_batchsize: true
  
  # パディング除去
  use_remove_padding: true
  
  # CPU オフロード
  cpu_offload: false
  
  # ZeRO 最適化
  zero_stage: 2
  
  # アクティベーションオフロード
  activation_offload: false
```

## 推論設定

### vLLM 設定

```yaml
inference:
  backend: vllm
  
  # GPU メモリ使用率
  gpu_memory_utilization: 0.7
  
  # 並列設定
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  
  # バッチ設定
  max_num_seqs: 256
  max_num_batched_tokens: 8192
  
  # 生成パラメータ
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  
  # その他
  trust_remote_code: true
  enforce_eager: false
```

### SGLang 設定

```yaml
inference:
  backend: sglang
  
  # メモリ設定
  mem_fraction_static: 0.7
  
  # 並列設定
  tp_size: 1
  
  # バッチ設定
  max_running_requests: 256
  max_total_tokens: 8192
  
  # 生成パラメータ
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

## 分散設定

### 基本分散設定

```yaml
distributed:
  # バックエンド
  backend: nccl  # nccl, gloo
  
  # 初期化方法
  init_method: env://
  
  # タイムアウト
  timeout: 1800
  
  # ランク情報
  world_size: 8
  rank: 0
```

### 並列化設定

```yaml
parallelism:
  # データ並列
  data_parallel_size: 4
  
  # テンソル並列
  tensor_parallel_size: 2
  
  # パイプライン並列
  pipeline_parallel_size: 1
  
  # シーケンス並列
  sequence_parallel: false
  ulysses_degree: 1
```

## 報酬設定

### ルールベース報酬

```yaml
reward:
  type: rule_based
  class_name: GSM8KReward
  
  parameters:
    correct_reward: 1.0
    incorrect_reward: 0.0
    partial_credit: false
```

### モデルベース報酬

```yaml
reward:
  type: model_based
  model_path: "reward_models/math_reward"
  
  parameters:
    batch_size: 32
    device: cuda
    normalize_scores: true
```

### 複合報酬

```yaml
reward:
  type: multi_reward
  
  rewards:
    - name: correctness
      type: rule_based
      weight: 0.7
      class_name: CorrectnessReward
    
    - name: quality
      type: model_based
      weight: 0.3
      model_path: "quality_reward_model"
```

## 実験追跡

### Weights & Biases

```yaml
wandb:
  enabled: true
  project: "verl-experiments"
  name: "ppo-qwen2.5-7b"
  tags:
    - ppo
    - qwen
    - gsm8k
  
  config:
    log_model: true
    log_gradients: false
    log_parameters: false
```

### TensorBoard

```yaml
tensorboard:
  enabled: true
  log_dir: "logs/tensorboard"
  log_interval: 10
```

## 環境固有設定

### SLURM 設定

```yaml
slurm:
  job_name: "verl-training"
  partition: "gpu"
  nodes: 2
  ntasks_per_node: 4
  gres: "gpu:4"
  time: "24:00:00"
```

### Docker 設定

```yaml
docker:
  image: "verlai/verl:latest"
  volumes:
    - "/data:/workspace/data"
    - "/models:/workspace/models"
  environment:
    CUDA_VISIBLE_DEVICES: "0,1,2,3"
```

## 設定の継承と上書き

### ベース設定

```yaml
# base_config.yaml
defaults:
  - model: qwen2.5-7b
  - algorithm: ppo
  - data: gsm8k

model:
  torch_dtype: bfloat16

training:
  max_steps: 1000
```

### 実験固有設定

```yaml
# experiment_config.yaml
defaults:
  - base_config

# 上書き設定
training:
  max_steps: 2000
  learning_rate: 5e-7

wandb:
  name: "experiment-v2"
```

## 設定の検証

### 設定チェック

```python
from verl.config import ConfigValidator

validator = ConfigValidator()
errors = validator.validate("config.yaml")

if errors:
    for error in errors:
        print(f"エラー: {error}")
else:
    print("設定は有効です")
```

### 設定の可視化

```python
from verl.config import ConfigVisualizer

visualizer = ConfigVisualizer()
visualizer.show_config("config.yaml")
```

## ベストプラクティス

1. **モジュラー設計**: 機能ごとに設定を分割
2. **環境分離**: 開発・本番環境で設定を分ける
3. **バージョン管理**: 設定ファイルもバージョン管理
4. **ドキュメント**: 設定項目の説明を記載
5. **検証**: 設定の妥当性を定期的にチェック

## トラブルシューティング

### 一般的な問題

1. **パス設定エラー**: 相対パスと絶対パスの確認
2. **型エラー**: YAML の型指定の確認
3. **メモリ不足**: バッチサイズとメモリ設定の調整
4. **分散設定エラー**: ノード間通信の確認

### デバッグ方法

```yaml
# デバッグ設定
debug:
  enabled: true
  log_level: DEBUG
  profile_memory: true
  profile_time: true
```

## 次のステップ

- [PPO アルゴリズム](../algo/ppo.md)で具体的なアルゴリズム設定を学ぶ
- [パフォーマンスチューニング](../web_content/performance_tuning_guide.md)で最適化設定を確認
- [マルチノード設定](../start/multinode.md)で分散設定の詳細を学ぶ
