# マルチノード設定

複数のノードで verl を実行する方法について説明します。

## 概要

マルチノード設定により、大規模なモデルや大量のデータを効率的に処理できます。verl は以下のマルチノード構成をサポートしています：

- データ並列化
- モデル並列化
- パイプライン並列化
- ハイブリッド並列化

## 前提条件

- 複数のノードが同じネットワークに接続されている
- 各ノードで verl がインストールされている
- 共有ストレージまたはデータの同期機能がある
- SSH アクセスが設定されている

## 設定方法

### 1. ネットワーク設定

#### マスターノードの設定

```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
```

#### ワーカーノードの設定

```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=1  # ノードごとに異なる値
```

### 2. SLURM での実行

#### SLURM スクリプト例

```bash
#!/bin/bash
#SBATCH --job-name=verl-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# 環境変数の設定
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# verl の実行
srun python -m verl.trainer.ppo_trainer \
    --config config/multinode_ppo.yaml \
    --num_nodes $SLURM_NNODES \
    --num_gpus_per_node 4
```

### 3. 手動での実行

#### マスターノードでの実行

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    -m verl.trainer.ppo_trainer \
    --config config/multinode_ppo.yaml
```

#### ワーカーノードでの実行

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    -m verl.trainer.ppo_trainer \
    --config config/multinode_ppo.yaml
```

## 設定ファイル

### マルチノード用設定例

```yaml
# 並列化設定
parallelism:
  data_parallel_size: 4
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  
# 分散設定
distributed:
  backend: nccl
  init_method: env://
  timeout: 1800

# モデル設定
model:
  path: models/qwen2.5-70b
  trust_remote_code: true
  
# トレーニング設定
training:
  batch_size: 128
  micro_batch_size: 4
  gradient_accumulation_steps: 8
  
# 推論設定
inference:
  backend: vllm
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
```

## 並列化戦略

### データ並列化

```yaml
parallelism:
  data_parallel_size: 8
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
```

### テンソル並列化

```yaml
parallelism:
  data_parallel_size: 2
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
```

### パイプライン並列化

```yaml
parallelism:
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 2
```

## 監視とデバッグ

### ログの確認

```bash
# 各ノードのログを確認
tail -f outputs/multinode_ppo/logs/rank_*.log
```

### ネットワーク通信の確認

```bash
# NCCL テスト
python -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
print(f'Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}')
"
```

### GPU 使用率の監視

```bash
# 各ノードで実行
nvidia-smi -l 1
```

## パフォーマンス最適化

### ネットワーク最適化

```bash
# InfiniBand の設定
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Ethernet の設定
export NCCL_SOCKET_IFNAME=eth0
```

### メモリ最適化

```yaml
training:
  gradient_checkpointing: true
  cpu_offload: true
  zero_stage: 2
```

## トラブルシューティング

### 接続エラー

- ファイアウォール設定を確認
- ポートが開いているか確認
- ネットワーク接続を確認

### 同期エラー

- タイムアウト値を増加
- 初期化方法を確認
- プロセス数を確認

### メモリエラー

- バッチサイズを減らす
- CPU オフロードを有効化
- ZeRO 最適化を使用

## ベストプラクティス

### 設定の推奨事項

1. **小規模モデル（< 7B）**: データ並列化のみ
2. **中規模モデル（7B-70B）**: データ + テンソル並列化
3. **大規模モデル（> 70B）**: 全ての並列化手法を組み合わせ

### 監視の推奨事項

1. GPU 使用率を定期的に確認
2. ネットワーク帯域幅を監視
3. メモリ使用量を追跡
4. スループットを測定

### デバッグの推奨事項

1. 単一ノードで動作確認
2. 段階的にノード数を増加
3. ログレベルを詳細に設定
4. プロファイリングツールを使用

## 次のステップ

- [パフォーマンスチューニングガイド](../web_content/performance_tuning_guide.md)でさらなる最適化を学ぶ
- [高度な機能](../advance/)で専門的な設定を確認
- [トラブルシューティング](../faq.md)で一般的な問題の解決方法を学ぶ
