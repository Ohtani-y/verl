# PPO (Proximal Policy Optimization)

PPO は verl でサポートされている主要な強化学習アルゴリズムの一つです。

## 概要

Proximal Policy Optimization (PPO) は、ポリシー勾配法の一種で、安定性と効率性を両立した強化学習アルゴリズムです。verl では、大規模言語モデルの RLHF（Reinforcement Learning from Human Feedback）に最適化された PPO 実装を提供しています。

## 主な特徴

- **安定したトレーニング**: クリッピング機構により、ポリシーの急激な変化を防止
- **効率的な実装**: 大規模モデルに対応した分散トレーニング
- **柔軟な設定**: 様々なハイパーパラメータの調整が可能

## 基本的な使用方法

### 設定ファイル

```yaml
algorithm:
  name: ppo
  learning_rate: 1e-6
  batch_size: 32
  mini_batch_size: 8
  epochs: 3
  clip_range: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 1.0
```

### 実行例

```bash
python -m verl.trainer.ppo_trainer \
    --config config/ppo_config.yaml \
    --model_path models/qwen2.5-7b \
    --data_path data/gsm8k_processed
```

## ハイパーパラメータ

### 学習率 (learning_rate)

- **推奨値**: 1e-6 から 5e-6
- **説明**: ポリシーネットワークの学習率
- **調整指針**: 大きすぎると不安定、小さすぎると収束が遅い

### クリップ範囲 (clip_range)

- **推奨値**: 0.1 から 0.3
- **説明**: ポリシー比のクリッピング範囲
- **調整指針**: 安定性と学習速度のバランス

### バリュー損失係数 (value_loss_coef)

- **推奨値**: 0.5 から 1.0
- **説明**: 価値関数損失の重み
- **調整指針**: 価値関数の学習速度を調整

### エントロピー係数 (entropy_coef)

- **推奨値**: 0.01 から 0.1
- **説明**: 探索を促進するエントロピー正則化の重み
- **調整指針**: 探索と活用のバランス

## 高度な設定

### アドバンテージ正規化

```yaml
ppo:
  normalize_advantages: true
  advantage_estimator: gae
  gae_lambda: 0.95
```

### KL 発散制約

```yaml
ppo:
  kl_target: 0.01
  kl_coef: 0.1
  adaptive_kl: true
```

### グラディエントクリッピング

```yaml
ppo:
  max_grad_norm: 1.0
  clip_grad_value: null
```

## パフォーマンス最適化

### メモリ最適化

```yaml
training:
  gradient_checkpointing: true
  use_dynamic_batchsize: true
  offload_optimizer: true
```

### 分散トレーニング

```yaml
distributed:
  data_parallel_size: 4
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
```

## トラブルシューティング

### 収束しない場合

1. 学習率を下げる
2. クリップ範囲を調整
3. バッチサイズを増やす
4. データの品質を確認

### 不安定な学習

1. グラディエントクリッピングを強化
2. KL 発散制約を追加
3. エントロピー係数を調整

### メモリ不足

1. バッチサイズを減らす
2. グラディエントチェックポイントを有効化
3. オプティマイザオフロードを使用

## ベストプラクティス

1. **段階的な調整**: 一度に一つのパラメータを調整
2. **ログの監視**: 損失、報酬、KL 発散を定期的に確認
3. **早期停止**: 過学習を防ぐため適切なタイミングで停止
4. **実験管理**: WandB などで実験を追跡

## 参考文献

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
