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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["CheckpointConfig", "ProfileConfig", "BaseModelConfig"]


@dataclass
class CheckpointConfig(BaseConfig):
    """モデルチェックポイントの設定。

    BaseConfig からの継承により、dataclass 設定に omegaconf.DictConfig のようなインターフェースを提供します。

    Args:
        save_contents (list[str]): 保存されるチェックポイントに含める内容。
            オプション: 'model', 'optimizer', 'extra', 'hf_model'。
        load_contents (list[str]): チェックポイントから読み込む内容。デフォルトは save_contents と同じ。
        async_save (bool): チェックポイントを非同期で保存するかどうか。現在は Megatron でのみ実装されています。
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False


@dataclass
class ProfileConfig(BaseConfig):
    """プロファイリングの設定。

    BaseConfig からの継承により、dataclass 設定に omegaconf.DictConfig のようなインターフェースを提供します。

    Args:
        use_profile (bool): プロファイリングを有効にするかどうか。
        profile_ranks (Optional[list[int]]): プロファイリングするランクのリスト。None は全ランクを意味します。
        step_start (int): プロファイリングの開始ステップ。
        step_end (int): プロファイリングの終了ステップ。
        save_path (Optional[str]): プロファイリング結果を保存するパス。
    """

    use_profile: bool = False
    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class BaseModelConfig(BaseConfig):
    """モデルの基本設定。
    事前訓練されたモデルチェックポイントの読み込みと初期化のためのコア設定を含みます。

    Args:
        path (str): 事前訓練されたモデル重みへのパス。
        tokenizer_path (Optional[str]): トークナイザーのパス（設定されていない場合は actor のモデルパスがデフォルト）。
        override_config (dict): Hugging Face 設定のオーバーライド。
        external_lib (Optional[str]): 外部モデル実装（オプション）。
        trust_remote_code (bool): Hugging Face モデルからのリモートコードを信頼するかどうか。
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False
