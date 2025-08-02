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
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import BaseModelConfig, CheckpointConfig
from verl.utils.profiler import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .optimizer import OptimizerConfig

__all__ = ["CriticConfig", "FSDPCriticConfig", "McoreCriticConfig", "FSDPCriticModelCfg"]


@dataclass
class CriticConfig(BaseConfig):
    """Critic モデルトレーニングの設定。

    BaseConfig からの継承により、dataclass 設定に omegaconf.DictConfig のようなインターフェースを提供します。

    Args:
        strategy (str): Critic モデルトレーニングに使用する戦略 (fsdp, fsdp2, megatron)。
        ppo_micro_batch_size_per_gpu (int): ローカル GPU あたりのマイクロバッチサイズ。
        rollout_n (int): 更新あたりのロールアウト数 (actor rollout_n をミラー)。
        optim (Dict[str, Any]): lr、weight_decay などを含むオプティマイザ設定。
        model (Dict[str, Any]): path、tokenizer_path などを含むモデル設定。
        ppo_mini_batch_size (int): 更新あたりの PPO ミニバッチサイズ。
        ppo_micro_batch_size (Optional[int]): グローバルマイクロバッチサイズ (非推奨)。
        use_dynamic_bsz (bool): 実行時にバッチサイズを自動調整するかどうか。
        ppo_max_token_len_per_gpu (int): 1つの PPO バッチでの GPU あたりの最大トークン数。
        forward_max_token_len_per_gpu (int): フォワードパスでの GPU あたりの最大トークン長。
        ppo_epochs (int): バッチあたりの PPO エポック数。
        shuffle (bool): PPO エポック間でトレーニングデータをシャッフルするかどうか。
        cliprange_value (float): PPO 価値関数のクリッピング範囲。
        loss_agg_mode (str): 損失集約モード。
        checkpoint (Dict[str, Any]): チェックポイント設定。
        profiler (Dict[str, Any]): プロファイラ設定。
        enable (Optional[bool]): Critic を有効にするかどうか。
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_micro_batch_size_per_gpu",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
    }

    strategy: str = MISSING
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    enable: Optional[bool] = None
    rollout_n: int = 1
    ppo_mini_batch_size: int = 1
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 32768
    forward_max_token_len_per_gpu: int = 32768
    ppo_epochs: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    ppo_micro_batch_size: Optional[int] = None
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Critic 設定パラメータを検証します。"""
        assert self.strategy != MISSING
        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.ppo_micro_batch_size, self.ppo_micro_batch_size_per_gpu, "critic")

            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"[critic] ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """実行時パラメータで Critic 設定を検証します。

        Args:
            n_gpus: 利用可能な GPU の総数
            train_batch_size: データ設定からのトレーニングバッチサイズ
        """
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"critic.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        param = "micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreCriticConfig(CriticConfig):
    """Configuration for Megatron-based critic model training.

    The inheritance from CriticConfig provides all base critic configuration plus Megatron-specific settings.

    Args:
        nccl_timeout (int): NCCL timeout in seconds for distributed operations.
        megatron (Dict[str, Any]): Megatron-specific parallelism settings.
        load_weight (bool): Whether to load initial weights.
        data_loader_seed (Optional[int]): Seed for data loader.
    """

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    load_weight: bool = True
    data_loader_seed: Optional[int] = None

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate Megatron critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)


@dataclass
class FSDPCriticConfig(CriticConfig):
    """Configuration for FSDP-based critic model training.

    The inheritance from CriticConfig provides all base critic configuration plus FSDP-specific settings.

    Args:
        forward_micro_batch_size (int): Forward-only batch size during inference (global).
        forward_micro_batch_size_per_gpu (int): Forward-only batch size during inference (per GPU).
        ulysses_sequence_parallel_size (int): Sequence parallelism size for Ulysses-style model parallelism.
        grad_clip (float): Gradient clipping for critic updates.
    """

    _mutable_fields = CriticConfig._mutable_fields | {
        "forward_micro_batch_size",
        "forward_micro_batch_size_per_gpu",
    }

    strategy: str = "fsdp"
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0

    def __post_init__(self):
        """Validate FSDP critic configuration parameters."""
        super().__post_init__()

        if self.strategy in {"fsdp", "fsdp2"}:
            if self.ulysses_sequence_parallel_size > 1:
                if not self.model.get("use_remove_padding", False):
                    raise ValueError(
                        "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate FSDP critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)

        if not self.use_dynamic_bsz:
            sp_size = self.ulysses_sequence_parallel_size
            if self.ppo_micro_batch_size is not None:
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"critic.ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )


@dataclass
class FSDPCriticModelCfg(BaseModelConfig):
    """FSDP-enabled critic model configuration.
    Inherits base critic settings and adds distributed-memory and LoRA options.

    Args:
        use_shm (bool): Whether to use shared memory for loading the model.
        enable_activation_offload (bool): Offload activations to CPU to reduce GPU memory usage.
        use_remove_padding (bool): Use remove-padding optimization (saves compute).
        enable_gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency.
        fsdp_config (FSDPEngineConfig): FSDP-specific configuration block.
        lora_rank (int): Set to positive value to enable LoRA (e.g., 32).
        lora_alpha (int): LoRA scaling factor.
        target_modules (Union[str, List[str]]): LoRA target modules: "all-linear" or list of layer names.
    """

    use_shm: bool = False
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = True
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str | list[str] = "all-linear"
