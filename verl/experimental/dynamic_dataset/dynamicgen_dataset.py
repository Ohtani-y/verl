# Copyright 2025 Amazon.com Inc and/or its affiliates
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
トレーニングのイテレーション間で動的データ生成戦略を可能にするデータセットクラス。
このクラスは RLHFDataset を拡張し、AbstractDataGen インスタンスを使用してデータを生成します。

これは、proposer モデルがロールアウトデータに基づいて新しいタスクを生成する設定で特に有用です。
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.utils.dataset import RLHFDataset
from verl.utils.import_utils import load_extern_type

logger = logging.getLogger(__name__)


class AbstractDataGenerator(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def generate(self, dataset: Dataset) -> datasets.Dataset:
        """
        generate メソッドはサブクラスで実装する必要があります。
        Args:
            dataset: 生成元となるデータセット。
        Returns:
            サブクラスで実装された処理済みデータまたは結果。
        """
        pass


class MockDataGenerator(AbstractDataGenerator):
    """
    最初のデータポイントのみを再追加する何もしないデータ生成クラス。
    このクラスはプレースホルダーとテストに有用です。
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return dataset.dataframe.select([0])


class DynamicGenDataset(RLHFDataset):
    """
    データ生成戦略を使用してデータを処理するデータセットクラス。
    このクラスは RLHFDataset を拡張し、AbstractDataGen インスタンスを使用してデータを生成します。
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)
        self.datagen: AbstractDataGenerator = config.datagen
        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )
        datagen_cls = load_extern_type(config.datagen.path, config.datagen.name)

        abs_cls = AbstractDataGenerator
        if not issubclass(datagen_cls, abs_cls):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + " must inherit from {abs_cls}"
            )

        self.data_generator = datagen_cls(config.datagen)
        self.on_batch_end()

    def append_dataframe(self, new_dataframe: datasets.Dataset):
        new_dataframe = self.maybe_filter_out_long_prompts(new_dataframe)
        self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])

        logger.info(f"new dataset len: {len(self.dataframe)}")

    def on_batch_end(self, batch: DataProto) -> None:
        """
        提供されたデータ生成戦略を使用してデータを生成します。
        注意: このメソッドは各トレーニングバッチ後にデータセットを変更することを意図しています。
        """
        new_data = self.data_generator.generate(self)
        self.append_dataframe(new_data)
