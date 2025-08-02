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

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = ["OptimizerConfig", "FSDPOptimizerConfig", "McoreOptimizerConfig"]


@dataclass
class OptimizerConfig(BaseConfig):
    """基本オプティマイザ設定。

    Args:
        lr (float): 学習率。必須で指定する必要があります。
        lr_warmup_steps_ratio (float): ウォームアップステップ比率；総ステップ数は実行時に注入されます。
        total_training_steps (int): 総トレーニングステップ数（実行時にオーバーライドする必要があります）。
        weight_decay (float): 重み減衰係数。
        lr_warmup_steps (Optional[int]): ウォームアップステップ数；None の場合は lr_warmup_steps_ratio に委譲されます。
    """

    lr: float = MISSING
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1

    def __post_init__(self):
        assert self.lr != MISSING


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """基本 OptimizerConfig を拡張した FSDP オプティマイザ設定。

    Args:
        lr (float): 学習率。
        min_lr_ratio (Optional[float]): cosine スケジュールの最小学習率比率。
        warmup_style (str): 学習率ウォームアップスタイル："constant" または "cosine"。
        num_cycles (float): 学習率スケジュールでの cosine サイクル数。
    """

    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    num_cycles: float = 0.5

    def __post_init__(self):
        assert self.warmup_style in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """基本 OptimizerConfig を拡張した Mcore オプティマイザ設定。

    Args:
        optimizer (str): オプティマイザ名；デフォルトは "adam"。
        lr (float): 学習率。
        clip_grad (float): 勾配クリッピングのノルム。
        lr_warmup_init (float): ウォームアップの初期学習率；デフォルトは 0.0。
        lr_decay_steps (Optional[int]): 減衰ステップ数。
        lr_decay_style (str): 学習率減衰スタイル："constant"、"linear"、"cosine"、または "inverse_square_root"。
        min_lr (float): 最小学習率。
        weight_decay_incr_style (str): 重み減衰増分スタイル："constant" または "cosine"。
        lr_wsd_decay_style (str): 重み標準偏差減衰スタイル："constant"、"exponential"、または "cosine"。
        lr_wsd_decay_steps (Optional[int]): 重み標準偏差減衰のステップ数。
        use_checkpoint_opt_param_scheduler (bool): チェックポイントオプティマイザパラメータスケジューラを使用するかどうか。
    """

    optimizer: str = "adam"
    clip_grad: float = 1.0
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
