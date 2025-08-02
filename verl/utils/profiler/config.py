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

from verl.base_config import BaseConfig


@dataclass
class ProfilerConfig(BaseConfig):
    """ワーカープロファイラー設定。現在は Nsight system profiler のみサポート。

    BaseConfig からの継承により、データクラス設定に omegaconf.DictConfig のようなインターフェースを提供。

    Args:
        discrete (bool): True の場合、各タスクが独自のデータベースを持つ。False の場合、1つのトレーニングステップ内の
          すべてのタスクが1つのデータベースを共有。
        all_ranks (bool): すべてのランクをプロファイルするかどうか。
        ranks (list[int]): プロファイルされるランク。デフォルトは []。
    """

    discrete: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            discrete=self.discrete or other.discrete,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            discrete=self.discrete and other.discrete,
        )

    def __post_init__(self) -> None:
        """設定検証ロジックをここに記述"""
        assert isinstance(self.ranks, set | list | tuple), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )
