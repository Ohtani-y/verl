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
報酬モデルのベースクラス
"""

from abc import ABC, abstractmethod

from verl import DataProto


class BasePPORewardModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        """input_ids を与えて報酬を計算します。transformers は形状
           [batch_size, sequence_length] のテンソルを出力し、[EOS] マスクの値を収集する必要があります。

        Args:
            data: "input_ids"、"attention_mask"、"position_ids" のキーを含む必要があります。
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: "reward" を含むデータパスプロトコル。[EOS] 位置のみが報酬を含みます。
            他の位置はゼロ報酬である必要があります。将来的に密な報酬を使用する場合は
            これが変更される可能性があることに注意してください。そのため、一般的なケースのための
            インターフェースを残しています。
            - reward: [batch_size, sequence_length]。

        """
        pass
