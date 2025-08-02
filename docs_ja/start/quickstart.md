# クイックスタートガイド

このガイドでは、GSM8K データセットを使用した PPO トレーニングの実行方法を説明します。

## 前提条件

- verl がインストールされていること（[インストールガイド](install.md)を参照）
- GPU が利用可能であること
- 十分なディスク容量があること（モデルとデータセット用）

## ステップ 1: データセットの準備

### GSM8K データセットのダウンロード

```bash
# データディレクトリの作成
mkdir -p data/gsm8k

# GSM8K データセットのダウンロード
python -m verl.utils.download_dataset \
    --dataset gsm8k \
    --output_dir data/gsm8k
```

### データの前処理

```bash
python -m verl.data.prepare_gsm8k \
    --input_dir data/gsm8k \
    --output_dir data/gsm8k_processed
```

## ステップ 2: モデルのダウンロード

### ベースモデルのダウンロード

```bash
# Qwen2.5-7B モデルのダウンロード
python -m verl.utils.download_model \
    --model_name Qwen/Qwen2.5-7B \
    --output_dir models/qwen2.5-7b
```

### SFT モデルの準備（オプション）

事前にファインチューニングされたモデルがある場合は、そのパスを指定してください。

## ステップ 3: 設定ファイルの準備

### 基本設定ファイル

`config/ppo_gsm8k.yaml` を作成：

```yaml
# モデル設定
model:
  path: models/qwen2.5-7b
  trust_remote_code: true

# データ設定
data:
  train_files: data/gsm8k_processed/train.jsonl
  val_files: data/gsm8k_processed/val.jsonl
  max_length: 2048

# PPO 設定
ppo:
  learning_rate: 1e-6
  batch_size: 32
  mini_batch_size: 8
  epochs: 3
  clip_range: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01

# トレーニング設定
training:
  max_steps: 1000
  save_steps: 100
  eval_steps: 50
  logging_steps: 10
  output_dir: outputs/ppo_gsm8k

# 推論設定
inference:
  backend: vllm
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9

# 報酬設定
reward:
  type: gsm8k_math_reward
  correct_reward: 1.0
  incorrect_reward: 0.0
```

## ステップ 4: PPO トレーニングの実行

### 単一 GPU での実行

```bash
python -m verl.trainer.ppo_trainer \
    --config config/ppo_gsm8k.yaml \
    --num_gpus 1
```

### マルチ GPU での実行

```bash
torchrun --nproc_per_node=4 \
    -m verl.trainer.ppo_trainer \
    --config config/ppo_gsm8k.yaml \
    --num_gpus 4
```

### SLURM での実行

```bash
sbatch scripts/run_ppo_gsm8k.sh
```

## ステップ 5: トレーニングの監視

### ログの確認

```bash
# リアルタイムログの表示
tail -f outputs/ppo_gsm8k/train.log

# TensorBoard の起動
tensorboard --logdir outputs/ppo_gsm8k/logs
```

### Weights & Biases での監視

設定ファイルに以下を追加：

```yaml
wandb:
  project: verl-gsm8k
  name: ppo-qwen2.5-7b
  tags: [ppo, gsm8k, qwen]
```

## ステップ 6: 結果の評価

### モデルの評価

```bash
python -m verl.eval.evaluate_gsm8k \
    --model_path outputs/ppo_gsm8k/final_model \
    --test_file data/gsm8k_processed/test.jsonl \
    --output_file outputs/ppo_gsm8k/eval_results.json
```

### 結果の分析

```bash
python -m verl.utils.analyze_results \
    --results_file outputs/ppo_gsm8k/eval_results.json
```

## トレーニングログの例

トレーニング中に以下のようなログが出力されます：

```
Step 10/1000 | Loss: 2.345 | Reward: 0.123 | KL: 0.045
Step 20/1000 | Loss: 2.234 | Reward: 0.156 | KL: 0.043
Step 30/1000 | Loss: 2.123 | Reward: 0.189 | KL: 0.041
...
```

## チェックポイントの管理

### チェックポイントの保存

チェックポイントは自動的に `outputs/ppo_gsm8k/checkpoints/` に保存されます。

### チェックポイントからの再開

```bash
python -m verl.trainer.ppo_trainer \
    --config config/ppo_gsm8k.yaml \
    --resume_from outputs/ppo_gsm8k/checkpoints/step_500
```

## 高度な設定

### メモリ最適化

```yaml
training:
  gradient_checkpointing: true
  use_dynamic_batchsize: true
  use_remove_padding: true
```

### パフォーマンス最適化

```yaml
inference:
  gpu_memory_utilization: 0.7
  max_num_seqs: 256
  tensor_parallel_size: 1
```

## トラブルシューティング

### OOM エラー

- バッチサイズを減らす
- グラディエントチェックポイントを有効化
- GPU メモリ使用率を下げる

### 低スループット

- バッチサイズを増やす
- 動的バッチサイズを有効化
- テンソル並列サイズを調整

### 収束しない

- 学習率を調整
- PPO パラメータを調整
- データの品質を確認

## 次のステップ

- [プログラミングガイド](../hybrid_flow.md)でより詳細な使用方法を学ぶ
- [パフォーマンスチューニングガイド](../web_content/performance_tuning_guide.md)でパフォーマンスを最適化
- [アルゴリズムガイド](../algo/)で他の RL アルゴリズムを試す
