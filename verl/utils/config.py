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

from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    OmegaConf DictConfig を dataclass に変換します。

    Args:
        config: 変換する OmegaConf DictConfig または dict。
        dataclass_type: 変換先の dataclass 型。dataclass_type が None の場合、
            DictConfig は hydra.instantiate API 経由でインスタンス化するために _target_ を含む必要があります。

    Returns:
        dataclass インスタンス。
    """
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "dataclass_type が提供されていない場合、config は _target_ を含む必要があります。 "
            "例については trainer/config/ppo_trainer.yaml の algorithm セクションを参照してください。 "
            f"取得した config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} は dataclass である必要があります")
    cfg = OmegaConf.create(config)  # dict の場合に備えて
    if "_target_" in cfg:
        cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)
