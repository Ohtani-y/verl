# パフォーマンスチューニングガイド

最終更新日：2025年7月17日

著者：Guangming Sheng、Jiali Zheng

このセクションでは、verl のすべてのステージのパフォーマンスを調整する方法について説明します。以下の内容を含みます：

1. ロールアウト生成スループット
2. シーケンスパッキング（`use_remove_padding=True`）を有効にしてデータパッキングとパディング除去を行う
3. 前方および後方計算のバッチサイズ調整
4. より高いスループットのための `use_dynamic_batchsize=True` を有効にする
5. 長いコンテキストトレーニングのための Ulysses Sequence Parallel を利用する
6. SFT パフォーマンス最適化のための LigerKernel
7. FSDP トレーニングバックエンドでの前方プリフェッチ
8. ログからのエントロピー計算のメモリ最適化

## ロールアウト生成チューニング

verl は現在、2つのロールアウトバックエンドをサポートしています：vLLM と TGI（SGLang サポートは近日公開予定）。

vLLM ベースのロールアウトを調整するための主要な要因を以下に示します。調整前に、ロールアウト統計がログに記録されるように `actor_rollout_ref.rollout.disable_log_stats=False` を設定することをお勧めします。

### GPU メモリ使用率の増加

`gpu_memory_utilization` を増加させます。

- vLLM v0.7.0 以降では、vLLM インスタンスは合計メモリの `gpu_memory_utilization` のみを使用します。
- SGLang の場合、これはモデルの重みや KV キャッシュなどの静的メモリに使用される空き GPU メモリの割合です。ただし、残りの（1 - `gpu_memory_utilization`）メモリは推論に使用されます。

ただし、モデルパラメータとオプティマイザの状態がオフロードされていない場合、高すぎる割合を使用すると OOM につながる可能性があります。0.5 から 0.7 の間の値は、高いスループットを実現し、OOM を回避するための良いバランスを取ることが多いです。

注意：`gpu_memory_utilization` の定義は推論エンジン間で異なるため、あるエンジンで機能する値が別のエンジンで OOM を引き起こす可能性があります。

### バッチサイズの調整

`max_num_seqs` または `max_num_batched_tokens` を調整します。GPU キャッシュ使用率がログで比較的低い場合、`max_num_seqs` または `max_num_batched_tokens` を増加させることができます。デコーディングステージでより多くの同時リクエストを可能にし、より高いスループットを実現するために `max_num_seqs` を設定することをお勧めします。

### テンソル並列サイズの最適化

より高いスループットのために小さな `tensor_parallel_size` を使用します。GPU リソースが許可する場合、より小さなテンソル並列サイズがより多くの vLLM レプリカを生成します。データ並列化（DP）により、vLLM レプリカ間でより良いスケーラビリティが得られます。

## シーケンスパッキングの有効化

シーケンスパッキングとパディング除去を有効にするには、`use_remove_padding=True` を設定します。

シーケンスパッキングにより、データパッキングとパディング除去が可能になります。これにより、以下の利点があります：

- メモリ使用量の削減
- 計算効率の向上
- より高いスループットの実現

## バッチサイズ調整

前方および後方計算のバッチサイズを調整することで、パフォーマンスを最適化できます。

### 動的バッチサイズの使用

動的バッチサイズを有効にするには、`use_dynamic_batchsize=True` を設定します。

動的バッチサイズにより、GPU リソースをより効率的に利用し、スループットを向上させることができます。GPU メモリが許可する限り、バッチサイズが自動的に調整されます。

## Ulysses Sequence Parallel の利用

長いコンテキストトレーニングのために Ulysses Sequence Parallel を利用できます。

Ulysses Sequence Parallel は、長いシーケンスを複数の GPU に分散することで、メモリ使用量を削減し、長いコンテキストでのトレーニングを可能にします。

設定例：
```yaml
model:
  sequence_parallel: true
  ulysses_degree: 2
```

## LigerKernel による SFT 最適化

SFT パフォーマンス最適化のために LigerKernel を使用できます。

LigerKernel は、メモリ効率的なカーネル実装を提供し、SFT のパフォーマンスを向上させます。

設定例：
```yaml
model:
  use_liger_kernel: true
```

## FSDP での前方プリフェッチ

FSDP トレーニングバックエンドで前方プリフェッチを有効にすることで、パフォーマンスを向上させることができます。

前方プリフェッチにより、次のレイヤーのパラメータを事前に読み込み、計算とデータ転送のオーバーラップを実現します。

## メモリ最適化

### エントロピー計算の最適化

ログからのエントロピー計算のメモリ最適化により、メモリ使用量を削減できます。

### GPU キャッシュ最適化

GPU キャッシュの使用率を監視し、適切に調整することで、パフォーマンスを最適化できます。

## 推奨設定

### 高スループット設定

```yaml
actor_rollout_ref:
  rollout:
    disable_log_stats: false
    gpu_memory_utilization: 0.7
    max_num_seqs: 256
    tensor_parallel_size: 1

training:
  use_remove_padding: true
  use_dynamic_batchsize: true
  
model:
  sequence_parallel: true
  use_liger_kernel: true
```

### メモリ効率設定

```yaml
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.5
    max_num_seqs: 128
    
training:
  use_remove_padding: true
  gradient_checkpointing: true
  
model:
  use_liger_kernel: true
```

## トラブルシューティング

### OOM エラー

- `gpu_memory_utilization` を下げる
- `max_num_seqs` を減らす
- `gradient_checkpointing` を有効にする

### 低スループット

- `gpu_memory_utilization` を上げる
- `max_num_seqs` を増やす
- `tensor_parallel_size` を下げる
- `use_dynamic_batchsize=True` を設定

### 長いコンテキストでの問題

- Ulysses Sequence Parallel を有効にする
- シーケンスパッキングを使用する
- メモリ効率的なアテンション機構を使用する

## パフォーマンス監視

パフォーマンスを監視するために、以下のメトリクスを確認してください：

- GPU 使用率
- メモリ使用率
- スループット（tokens/sec）
- レイテンシ
- キャッシュヒット率

これらのメトリクスを定期的に監視し、設定を調整することで、最適なパフォーマンスを実現できます。
