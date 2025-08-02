# データセット形式
## RLHF データセット
すべてのデータソースを単一の parquet ファイルに結合します。プロンプトを直接チャット形式に整理することで、マルチターンチャットを簡単に組み込むことができます。プロンプトには、モデルが特定の形式で回答を出力するよう指導する指示従順テキストを追加し、回答を抽出できるようにします。

数学問題
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```
