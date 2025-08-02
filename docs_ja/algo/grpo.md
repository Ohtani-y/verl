# GRPO (Group Relative Policy Optimization)

GRPO は verl でサポートされている高効率な強化学習アルゴリズムです。

## 概要

Group Relative Policy Optimization (GRPO) は、グループベースの相対的な最適化を行う強化学習アルゴリズムです。従来の PPO と比較して、より効率的で安定したトレーニングを実現します。

## 主な特徴

- **高効率**: PPO と比較して高いサンプル効率
- **安定性**: グループベースの正規化により安定したトレーニング
- **スケーラビリティ**: 大規模モデルでの優れたパフォーマンス

## 基本的な使用方法

### 設定ファイル

```yaml
algorithm:
  name: grpo
  learning_rate: 1e-6
  batch_size: 64
  mini_batch_size: 16
  epochs: 3
  group_size: 8
  baseline_type: group_mean
  temperature: 1.0
```

### 実行例

```bash
python -m verl.trainer.grpo_trainer \
    --config config/grpo_config.yaml \
    --model_path models/qwen2.5-7b \
    --data_path data/gsm8k_processed
```

## ハイパーパラメータ

### グループサイズ (group_size)

- **推奨値**: 4 から 16
- **説明**: 相対的な比較を行うグループのサイズ
- **調整指針**: 大きいほど安定、小さいほど効率的

### ベースライン種別 (baseline_type)

- **選択肢**: group_mean, group_median, exponential_moving_average
- **説明**: 報酬のベースライン計算方法
- **推奨**: group_mean（最も安定）

### 温度パラメータ (temperature)

- **推奨値**: 0.5 から 2.0
- **説明**: ソフトマックス分布の温度
- **調整指針**: 探索と活用のバランス

## 高度な設定

### 適応的グループサイズ

```yaml
grpo:
  adaptive_group_size: true
  min_group_size: 4
  max_group_size: 16
  group_size_schedule: linear
```

### 報酬正規化

```yaml
grpo:
  reward_normalization: true
  reward_scaling: 1.0
  reward_clipping: 10.0
```

### 勾配最適化

```yaml
grpo:
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  optimizer_type: adamw
```

## パフォーマンス比較

| アルゴリズム | サンプル効率 | 安定性 | 計算効率 |
|-------------|-------------|--------|----------|
| PPO         | 標準        | 高     | 標準     |
| GRPO        | 高          | 高     | 高       |
| REINFORCE   | 低          | 低     | 高       |

## 実装の詳細

### グループ分割

```python
def create_groups(responses, group_size):
    """レスポンスをグループに分割"""
    num_groups = len(responses) // group_size
    groups = []
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        groups.append(responses[start_idx:end_idx])
    return groups
```

### 相対的報酬計算

```python
def compute_relative_rewards(group_rewards, baseline_type):
    """グループ内での相対的報酬を計算"""
    if baseline_type == "group_mean":
        baseline = torch.mean(group_rewards)
    elif baseline_type == "group_median":
        baseline = torch.median(group_rewards)
    
    relative_rewards = group_rewards - baseline
    return relative_rewards
```

## 使用例

### 数学問題解決

```yaml
task:
  name: math_reasoning
  dataset: gsm8k
  
grpo:
  group_size: 8
  baseline_type: group_mean
  temperature: 1.0
  
reward:
  type: math_correctness
  partial_credit: true
```

### コード生成

```yaml
task:
  name: code_generation
  dataset: humaneval
  
grpo:
  group_size: 4
  baseline_type: group_median
  temperature: 0.8
  
reward:
  type: code_execution
  timeout: 10
```

## トラブルシューティング

### 収束が遅い場合

1. グループサイズを調整
2. 学習率を上げる
3. 温度パラメータを調整

### 不安定な学習

1. ベースライン種別を変更
2. 報酬正規化を有効化
3. グラディエントクリッピングを強化

### メモリ使用量が多い場合

1. グループサイズを減らす
2. バッチサイズを調整
3. グラディエント蓄積を使用

## ベストプラクティス

1. **適切なグループサイズ**: タスクの複雑さに応じて調整
2. **ベースライン選択**: データの分布に応じて最適な方法を選択
3. **温度調整**: 探索と活用のバランスを考慮
4. **実験追跡**: 複数の設定を比較検証

## 参考文献

- [Group Relative Policy Optimization for Reinforcement Learning](https://arxiv.org/abs/2306.09683)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
