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

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """KL制御の設定。

    BaseConfigからの継承により、dataclass設定にomegaconf.DictConfigのようなインターフェースを提供します。

    Args:
        type (str): KL制御のタイプ。"fixed"または"adaptive"を指定可能。
        kl_coef (float): KLペナルティの初期係数。
        horizon (int): 適応コントローラーのホライゾン値。
        target_kl (float): 適応コントローラーの目標KL発散。
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """フィルターグループの設定（DAPOとEntropyで使用）。

    BaseConfigからの継承により、dataclass設定にomegaconf.DictConfigのようなインターフェースを提供します。

    Args:
        enable (bool): フィルターグループを有効にするかどうか。
        metric (Optional[str]): フィルタリングに使用するメトリック："acc"、"score"、"seq_reward"、"seq_final_reward"など。
        max_num_gen_batches (int): 非正の値は上限なしを意味します。
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class AlgoConfig(BaseConfig):
    """アルゴリズムの設定。

    BaseConfigからの継承により、dataclass設定にomegaconf.DictConfigのようなインターフェースを提供します。

    Args:
        gamma (float): 将来の報酬の割引係数。
        lam (float): GAE推定器におけるバイアスと分散のトレードオフ。
        adv_estimator (str): アドバンテージ推定器のタイプ："gae"、"grpo"、"reinforce_plus_plus"など。
        norm_adv_by_std_in_grpo (bool): アドバンテージを標準偏差で正規化するかどうか（GRPO固有）。
        use_kl_in_reward (bool): 報酬内KLペナルティを有効にするかどうか。
        kl_penalty (str): KL発散の推定方法："kl"、"abs"、"mse"、"low_var_kl"、または"full"。
        kl_ctrl (KLControlConfig): KL制御の設定。
        use_pf_ppo (bool): 選好フィードバックPPOを有効にするかどうか。
        pf_ppo (dict[str, Any]): 選好フィードバックPPOの設定。
        filter_groups (Optional[FilterGroupsConfig]): フィルターグループの設定、DAPOとEntropyで使用
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
