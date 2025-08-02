# モデル
huggingface/transformers などの一般的なモデル動物園は、PyTorch ネイティブモデル並列化を使用する際に問題が生じます。vLLM の設計原則に従い、verl では packed inputs を使用したシンプルで並列化可能、高度に最適化されたモデルを保持しています。
## 新しい Huggingface モデルの追加
### ステップ 1: HF から verl にモデルファイルをコピー
- verl/models/hf の下に新しいファイルを追加
- huggingface/transformers/models から verl/models/hf にモデルファイルのみをコピー

### ステップ 2: packed inputs を使用するようにモデルファイルを修正
- 推論に関連するすべてのコード（kv cache）を削除
- 入力を以下のみを含むように修正
    - input_ids (total_nnz,)
    - cu_seqlens (total_nnz + 1,)
    - max_seqlen_in_batch: int
- これには causal mask を使用した flash attention が必要であることに注意

### ステップ 2.5: テストを追加
- このバージョンと huggingface バージョンを比較するテストを追加
- インフラストラクチャに従い、tests/models/hf にテストを追加

### ステップ 3: tensor parallelism を適用する関数を追加
- 以下に従ってください
    - https://pytorch.org/docs/stable/distributed.tensor.parallel.html
    - https://pytorch.org/tutorials/intermediate/TP_tutorial.html
- 一般的なコメント
    - PyTorch ネイティブの Tensor Parallelism は自動並列化ではありません。動作方法は、設定を使用してモデルパラメータと入力/出力の再分散方法を指定することです。これらの設定は、モデルの forward の前後で入力/出力の再分散を実行するフックとして登録されます。

### ステップ 4: data parallelism を適用する関数を追加
- FSDP2 API を使用してください
- デモはこちらを参照 https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L413

### ステップ 5: pipeline parallelism を適用する関数を追加
- PyTorch 2.4 で提供予定
- 現在は nightly バージョンでアルファ版のみ
- 詳細は torchtitan を確認

