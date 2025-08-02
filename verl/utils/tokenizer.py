# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""トークン化のためのユーティリティ。"""

import warnings

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """pad_token_id が None の場合、eos_token_id に設定する。

    Args:
        tokenizer (transformers.PreTrainedTokenizer): 設定するトークナイザー。

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """eos と pad トークンを適切に処理する Hugging Face 事前訓練済みトークナイザーを作成する。

    Args:

        name (str): トークナイザーの名前。
        correct_pad_token (bool): pad token id を修正するかどうか。
        correct_gemma2 (bool): gemma2 トークナイザーを修正するかどうか。

    Returns:

        transformers.PreTrainedTokenizer: 事前訓練済みトークナイザー。

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """マルチモーダルデータを処理する Hugging Face プロセッサーを作成する。

    Args:
        name_or_path (str): プロセッサーの名前。

    Returns:
        transformers.ProcessorMixin: 事前訓練済みプロセッサー。
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): サイレント失敗を避けるため、setup.py に transformer バージョン要件を追加した後、try-catch を削除すべき
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
