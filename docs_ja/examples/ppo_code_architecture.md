# PPO サンプルアーキテクチャ

verl での PPO 実装のコードアーキテクチャと構成要素について説明します。

## 概要

verl の PPO 実装は、モジュラー設計により柔軟性と拡張性を提供します。このドキュメントでは、PPO トレーニングの主要コンポーネントとその相互作用について詳しく説明します。

## アーキテクチャ概要

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Controller    │    │     Workers     │    │   Data Flow     │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ PPO Trainer │ │    │ │ Actor       │ │    │ │ Prompts     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Coordinator │ │    │ │ Critic      │ │    │ │ Responses   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Scheduler   │ │    │ │ Rollout     │ │    │ │ Rewards     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 主要コンポーネント

### 1. PPO Trainer

PPO トレーニングの中心的なコンポーネント：

```python
class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.actor_worker = self._create_actor_worker()
        self.critic_worker = self._create_critic_worker()
        self.rollout_worker = self._create_rollout_worker()
        self.reward_manager = self._create_reward_manager()
    
    def train(self):
        """メインのトレーニングループ"""
        for step in range(self.config.max_steps):
            # 1. ロールアウト生成
            rollout_data = self.generate_rollout()
            
            # 2. 報酬計算
            rewards = self.compute_rewards(rollout_data)
            
            # 3. アドバンテージ計算
            advantages = self.compute_advantages(rollout_data, rewards)
            
            # 4. PPO 更新
            self.update_policy(rollout_data, advantages)
            
            # 5. 評価とログ
            if step % self.config.eval_steps == 0:
                self.evaluate()
```

### 2. Actor Worker

ポリシーモデルの管理と更新：

```python
class ActorWorker:
    def __init__(self, model_config, training_config):
        self.model = self._load_model(model_config)
        self.optimizer = self._create_optimizer(training_config)
        self.scheduler = self._create_scheduler(training_config)
    
    def forward(self, input_ids, attention_mask):
        """前方パス"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits
    
    def compute_loss(self, logits, labels, old_logprobs, advantages):
        """PPO 損失の計算"""
        # 新しいログ確率の計算
        new_logprobs = self._compute_logprobs(logits, labels)
        
        # 比率の計算
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # クリップされた目的関数
        clipped_ratio = torch.clamp(
            ratio, 
            1 - self.config.clip_range, 
            1 + self.config.clip_range
        )
        
        # PPO 損失
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
    
    def update(self, batch_data):
        """モデルパラメータの更新"""
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(
            batch_data['logits'],
            batch_data['labels'],
            batch_data['old_logprobs'],
            batch_data['advantages']
        )
        
        loss.backward()
        
        # グラディエントクリッピング
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
```

### 3. Critic Worker

価値関数の管理と更新：

```python
class CriticWorker:
    def __init__(self, model_config, training_config):
        self.model = self._load_model(model_config)
        self.optimizer = self._create_optimizer(training_config)
    
    def forward(self, input_ids, attention_mask):
        """価値の予測"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits.squeeze(-1)  # [batch_size, seq_len]
    
    def compute_loss(self, values, returns):
        """価値関数の損失計算"""
        value_loss = F.mse_loss(values, returns)
        return value_loss
    
    def update(self, batch_data):
        """価値関数の更新"""
        self.optimizer.zero_grad()
        
        values = self.forward(
            batch_data['input_ids'],
            batch_data['attention_mask']
        )
        
        loss = self.compute_loss(values, batch_data['returns'])
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()
```

### 4. Rollout Worker

推論エンジンを使用したロールアウト生成：

```python
class RolloutWorker:
    def __init__(self, rollout_config):
        self.config = rollout_config
        self.engine = self._create_inference_engine()
    
    def generate(self, prompts):
        """プロンプトからレスポンスを生成"""
        responses = self.engine.generate(
            prompts=prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True
        )
        
        return responses
    
    def _create_inference_engine(self):
        """推論エンジンの作成"""
        if self.config.backend == "vllm":
            return VLLMEngine(self.config)
        elif self.config.backend == "sglang":
            return SGLangEngine(self.config)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
```

### 5. Reward Manager

報酬計算の管理：

```python
class RewardManager:
    def __init__(self, reward_config):
        self.config = reward_config
        self.reward_functions = self._load_reward_functions()
    
    def compute_rewards(self, prompts, responses):
        """報酬の計算"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            reward = 0.0
            
            # 複数の報酬関数を組み合わせ
            for reward_func, weight in self.reward_functions:
                partial_reward = reward_func.compute_reward(prompt, response)
                reward += weight * partial_reward
            
            rewards.append(reward)
        
        return torch.tensor(rewards)
    
    def _load_reward_functions(self):
        """報酬関数の読み込み"""
        functions = []
        
        for reward_config in self.config.rewards:
            if reward_config.type == "rule_based":
                func = self._create_rule_based_reward(reward_config)
            elif reward_config.type == "model_based":
                func = self._create_model_based_reward(reward_config)
            else:
                raise ValueError(f"Unknown reward type: {reward_config.type}")
            
            functions.append((func, reward_config.weight))
        
        return functions
```

## データフロー

### 1. ロールアウト生成フェーズ

```python
def generate_rollout(self):
    """ロールアウトデータの生成"""
    # 1. プロンプトの取得
    prompts = self.data_loader.get_batch()
    
    # 2. レスポンス生成
    responses = self.rollout_worker.generate(prompts)
    
    # 3. ログ確率の計算
    input_ids, attention_mask = self.tokenize(prompts, responses)
    with torch.no_grad():
        logits = self.actor_worker.forward(input_ids, attention_mask)
        old_logprobs = self.compute_logprobs(logits, input_ids)
    
    # 4. 価値の計算
    with torch.no_grad():
        values = self.critic_worker.forward(input_ids, attention_mask)
    
    return {
        'prompts': prompts,
        'responses': responses,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'old_logprobs': old_logprobs,
        'values': values
    }
```

### 2. 報酬計算フェーズ

```python
def compute_rewards(self, rollout_data):
    """報酬の計算"""
    rewards = self.reward_manager.compute_rewards(
        rollout_data['prompts'],
        rollout_data['responses']
    )
    
    return rewards
```

### 3. アドバンテージ計算フェーズ

```python
def compute_advantages(self, rollout_data, rewards):
    """GAE を使用したアドバンテージ計算"""
    values = rollout_data['values']
    
    # リターンの計算
    returns = self.compute_returns(rewards, values)
    
    # アドバンテージの計算
    advantages = returns - values
    
    # 正規化
    if self.config.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns
```

### 4. ポリシー更新フェーズ

```python
def update_policy(self, rollout_data, advantages):
    """PPO ポリシー更新"""
    returns = advantages + rollout_data['values']
    
    # データセットの作成
    dataset = PPODataset(
        input_ids=rollout_data['input_ids'],
        attention_mask=rollout_data['attention_mask'],
        old_logprobs=rollout_data['old_logprobs'],
        advantages=advantages,
        returns=returns
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=self.config.mini_batch_size,
        shuffle=True
    )
    
    # 複数エポックの更新
    for epoch in range(self.config.epochs):
        for batch in dataloader:
            # Actor の更新
            actor_loss = self.actor_worker.update(batch)
            
            # Critic の更新
            critic_loss = self.critic_worker.update(batch)
            
            # ログ記録
            self.log_metrics({
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'epoch': epoch
            })
```

## 分散トレーニング

### データ並列化

```python
class DistributedPPOTrainer(PPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        # 分散初期化
        dist.init_process_group(backend='nccl')
        
        # モデルの分散化
        self.actor_worker.model = DDP(self.actor_worker.model)
        self.critic_worker.model = DDP(self.critic_worker.model)
    
    def update_policy(self, rollout_data, advantages):
        """分散ポリシー更新"""
        # 全プロセスでデータを同期
        all_rollout_data = self.all_gather(rollout_data)
        all_advantages = self.all_gather(advantages)
        
        # 通常の更新処理
        super().update_policy(all_rollout_data, all_advantages)
```

### モデル並列化

```python
class ModelParallelPPOTrainer(PPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        # テンソル並列化の設定
        self.tp_size = config.tensor_parallel_size
        self.pp_size = config.pipeline_parallel_size
        
        # モデルの分割
        self.actor_worker.model = self.shard_model(
            self.actor_worker.model,
            self.tp_size,
            self.pp_size
        )
```

## メモリ最適化

### グラディエントチェックポイント

```python
class MemoryOptimizedActorWorker(ActorWorker):
    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)
        
        if training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def forward(self, input_ids, attention_mask):
        """メモリ効率的な前方パス"""
        if self.training and self.config.gradient_checkpointing:
            return checkpoint(
                self.model,
                input_ids,
                attention_mask,
                use_reentrant=False
            )
        else:
            return super().forward(input_ids, attention_mask)
```

### 動的バッチサイズ

```python
class DynamicBatchTrainer(PPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.current_batch_size = config.initial_batch_size
        self.max_batch_size = config.max_batch_size
    
    def generate_rollout(self):
        """動的バッチサイズでのロールアウト"""
        try:
            # 現在のバッチサイズで試行
            return self._generate_rollout_with_batch_size(
                self.current_batch_size
            )
        except torch.cuda.OutOfMemoryError:
            # OOM の場合はバッチサイズを減らす
            self.current_batch_size = max(
                self.current_batch_size // 2,
                1
            )
            torch.cuda.empty_cache()
            return self._generate_rollout_with_batch_size(
                self.current_batch_size
            )
```

## 監視とデバッグ

### メトリクス収集

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log_step_metrics(self, step, metrics):
        """ステップごとのメトリクス記録"""
        self.metrics['step'].append(step)
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_summary(self):
        """メトリクスの要約"""
        summary = {}
        
        for key, values in self.metrics.items():
            if key != 'step' and values:
                summary[f'{key}_mean'] = np.mean(values[-100:])  # 直近100ステップ
                summary[f'{key}_std'] = np.std(values[-100:])
        
        return summary
```

### プロファイリング

```python
class ProfiledPPOTrainer(PPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    
    def train_step(self):
        """プロファイリング付きトレーニングステップ"""
        with self.profiler:
            super().train_step()
        
        self.profiler.step()
```

## ベストプラクティス

1. **モジュラー設計**: 各コンポーネントを独立して開発・テスト
2. **エラーハンドリング**: 適切な例外処理とリカバリ機構
3. **ログ記録**: 詳細なログでデバッグを容易に
4. **メモリ管理**: 効率的なメモリ使用とガベージコレクション
5. **設定管理**: 柔軟で拡張可能な設定システム

## 次のステップ

- [設定ガイド](config.md)で詳細な設定方法を学ぶ
- [パフォーマンスチューニング](../web_content/performance_tuning_guide.md)で最適化手法を確認
- [分散トレーニング](../start/multinode.md)でスケールアップ方法を学ぶ
