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
from typing import Callable

import torch

from verl import DataProto

from ..base import BaseEngine, EngineRegistry


@EngineRegistry.register("megatron")
class MegatronEngine(BaseEngine):
    def __init__(self, config):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def train_mode(self):
        """
        エンジンとモデルをトレーニングモードに切り替えるためのコンテキストマネージャーエントリ。

        使用方法:
            with engine.train_mode():
        """
        raise NotImplementedError

    def eval_mode(self):
        """
        エンジンとモデルを評価モードに切り替えるためのコンテキストマネージャーエントリ。

        使用方法:
            with engine.eval_mode():
        """
        raise NotImplementedError

    def infer_batch(
        self,
        data: DataProto,
        post_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        データのミニバッチに対して推論を実行する。

        Args:
            data: 推論用の入力データ。通常はテンソルとメタデータを含む。
            post_fn: マイクロバッチと予測を入力として受け取り、
                     処理済み予測と出力の辞書を含むタプルを返す後処理関数。

        Returns:
            dict[str, torch.Tensor]: バッチ全体の予測を含む辞書。
        """
        raise NotImplementedError

    def train_batch(
        self,
        data: DataProto,
        loss_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        データのミニバッチに対してトレーニングステップを実行する。

        Args:
            data (DataProto): トレーニング用の入力データ。通常はテンソルとメタデータを含む。
            loss_fn (Callable): マイクロバッチと予測が与えられた時に損失とメトリクスを計算する関数。

        Returns:
            dict[str, torch.Tensor]: ミニバッチの集約されたトレーニングメトリクスを含む辞書。
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        新しい逆伝播を開始する前に、すべてのパラメータの勾配をゼロにする。
        """
        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

    def shard_data(self, data):
        """
        Shard or partition data for distributed training or parallel execution.

        Args:
            data: Data structure to be sharded across devices/workers.

        Returns:
            Sharded data in the same format as input.
        """
        raise NotImplementedError

    def unshard_data(self, data):
        """
        Reconstruct or gather sharded data back to a unified format.

        Args:
            data: Sharded data structure to reconstruct.

        Returns:
            Unsharded, combined data.
        """
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        raise NotImplementedError

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        raise NotImplementedError

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        raise NotImplementedError
