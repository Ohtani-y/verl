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
"""
Actor の基底クラス
"""

from abc import ABC, abstractmethod

import torch

from verl import DataProto

__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config):
        """PPO actor の基底クラス

        Args:
            config (DictConfig): PPOActor に渡される設定。DictConfig 型を期待しますが
                (https://omegaconf.readthedocs.io/)、一般的には任意の namedtuple でも可能です。
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """データのバッチが与えられた場合の logits を計算します。

        Args:
            data (DataProto): DataProto で表現されたデータのバッチ。```input_ids```、
                ```attention_mask```、```position_ids``` のキーを含む必要があります。

        Returns:
            DataProto: ```log_probs``` キーを含む DataProto


        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """DataProto のイテレータでポリシーを更新します

        Args:
            data (DataProto): ```make_minibatch_iterator``` によって返される
                DataProto のイテレータ

        Returns:
            Dict: 任意の内容を含む辞書。通常、モデル更新中の統計情報
            （```loss```、```grad_norm``` など）を含みます。

        """
        pass
