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

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["FSDPEngineConfig", "McoreEngineConfig"]


@dataclass
class McoreEngineConfig(BaseConfig):
    """Megatron 並列化の設定。

    BaseConfig からの継承により、データクラス設定に omegaconf.DictConfig のようなインターフェースを提供します。

    Args:
        param_offload (bool): パラメータを CPU にオフロードするかどうか。
        grad_offload (bool): 勾配を CPU にオフロードするかどうか。
        optimizer_offload (bool): オプティマイザの状態を CPU にオフロードするかどうか。
        tensor_model_parallel_size (int): テンソルモデル並列サイズ。
        expert_model_parallel_size (int): MoE モデル用のエキスパートモデル並列サイズ。
        expert_tensor_parallel_size (Optional[int]): MoE モデル用のエキスパートテンソル並列サイズ。
        pipeline_model_parallel_size (int): パイプラインモデル並列サイズ。
        virtual_pipeline_model_parallel_size (Optional[int]): インターリーブスケジューリング用の
            仮想パイプラインモデル並列サイズ。
        context_parallel_size (int): 長いシーケンス用のコンテキスト並列サイズ。
        sequence_parallel (bool): シーケンス並列化を有効にするかどうか。
        use_distributed_optimizer (bool): 分散オプティマイザを使用するかどうか。
        use_dist_checkpointing (bool): 分散チェックポイントを使用するかどうか。
        dist_checkpointing_path (Optional[str]): 分散チェックポイントのパス。
        seed (int): 再現性のためのランダムシード。
        override_ddp_config (dict[str, Any]): DDP の設定をオーバーライド。
        override_transformer_config (dict[str, Any]): transformer の設定をオーバーライド。
        use_mbridge (bool): 通信に MBridge を使用するかどうか。
    """

    # sequence_parallel は自動修正のため frozen フィールドとしてリストされていません
    _mutable_fields = BaseConfig._mutable_fields | {"sequence_parallel"}

    param_offload: bool = False
    grad_offload: bool = False
    optimizer_offload: bool = False
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    seed: int = 42
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = False

    def __post_init__(self) -> None:
        """設定検証ロジックをここに記述"""
        if self.tensor_model_parallel_size == 1:
            warnings.warn("TP サイズが 1 のため sequence parallel を false に設定", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class FSDPEngineConfig(BaseConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU, default False
        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True
        fsdp_size (int): FSDP group size. -1 means use all available GPUs.
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"
        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False
    """

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
