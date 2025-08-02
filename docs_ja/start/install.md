# インストール

verl のインストール方法について説明します。

## 要件

### Python バージョン

- Python 3.9 以上（Python 3.10 推奨）

### CUDA バージョン

- CUDA 11.8 以上（CUDA 12.1 推奨）

## バックエンドエンジンの選択

verl は複数のバックエンドエンジンをサポートしています：

### トレーニングバックエンド

- **FSDP**: PyTorch の Fully Sharded Data Parallel
- **FSDP2**: PyTorch の次世代 FSDP
- **Megatron-LM**: NVIDIA の大規模言語モデルトレーニングフレームワーク

### 推論バックエンド

- **vLLM**: 高性能 LLM 推論エンジン
- **SGLang**: 構造化生成言語エンジン
- **TGI**: Text Generation Inference

## インストール方法

### 1. 基本インストール

```bash
pip install verl
```

### 2. 開発版インストール

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

### 3. 特定のバックエンドでのインストール

#### vLLM バックエンド

```bash
pip install verl[vllm]
```

#### SGLang バックエンド

```bash
pip install verl[sglang]
```

#### Megatron-LM バックエンド

```bash
pip install verl[megatron]
```

#### 全バックエンド

```bash
pip install verl[all]
```

## Docker を使用したインストール

### 公式 Docker イメージ

```bash
docker pull verlai/verl:latest
```

### カスタム Docker イメージのビルド

```bash
git clone https://github.com/volcengine/verl.git
cd verl
docker build -t verl:custom .
```

## 環境設定

### 環境変数

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:/path/to/verl
```

### Weights & Biases（オプション）

```bash
pip install wandb
wandb login
```

## 依存関係の詳細

### 必須依存関係

- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.0.0
- accelerate >= 0.20.0

### オプション依存関係

- vllm >= 0.8.2（vLLM バックエンド用）
- sglang >= 0.4.0（SGLang バックエンド用）
- megatron-lm（Megatron バックエンド用）
- wandb（実験追跡用）
- tensorboard（ログ記録用）

## インストールの確認

```bash
python -c "import verl; print(verl.__version__)"
```

## トラブルシューティング

### CUDA エラー

CUDA バージョンが正しくインストールされていることを確認してください：

```bash
nvidia-smi
nvcc --version
```

### メモリエラー

GPU メモリが不足している場合は、以下を試してください：

- より小さなモデルを使用
- バッチサイズを減らす
- グラディエントチェックポイントを有効化

### 依存関係の競合

仮想環境を使用することを強く推奨します：

```bash
python -m venv verl_env
source verl_env/bin/activate
pip install verl
```

## AMD GPU サポート（ROCm）

AMD GPU での使用については、ROCm サポートドキュメントを参照してください。

### ROCm インストール

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install verl[rocm]
```

## 次のステップ

インストールが完了したら、[クイックスタートガイド](quickstart.md)に進んでください。
