## LiveCodeBench

### はじめに
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) は LLM のコーディング能力の包括的で汚染のない評価を提供します。特に、LiveCodeBench は LeetCode、AtCoder、CodeForces の3つの競技プラットフォームのコンテストから継続的に新しい問題を収集しています。

### 再現方法
私たちの評価は LiveCodeBench で見つかったバージョンに基づいています。
> **インストール**
```bash
# CUDA バージョンが 12.0 以上であることを確認してください。
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 謝辞
オープンソースコミュニティへの貢献に対して [LiveCodeBench](https://livecodebench.github.io/leaderboard.html) チームに感謝します。
