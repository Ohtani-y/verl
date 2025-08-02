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
Worker クラス
"""

import os
import socket
from dataclasses import dataclass

import ray

from verl.utils.device import get_torch_device, get_visible_devices_keyword

from .decorator import Dispatch, Execute, register


@dataclass
class DistRankInfo:
    tp_rank: int
    dp_rank: int
    pp_rank: int
    cp_rank: int


@dataclass
class DistGlobalInfo:
    tp_size: int
    dp_size: int
    pp_size: int
    cp_size: int


class WorkerHelper:
    @staticmethod
    def _get_node_ip():
        if os.getenv("WG_BACKEND", None) == "ray":
            return ray.util.get_node_ip_address()
        else:
            raise NotImplementedError("WG_BACKEND now just support ray mode.")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        return self._get_node_ip().strip("[]"), str(self._get_free_port())


class Worker(WorkerHelper):
    """分散トレーニングの初期化と設定を処理する分散ワーカー。

    このクラスはワーカーの初期化、設定を管理し、分散操作を実行するメソッドを提供します。
    通信設定、デバイス設定、ワーカーメタデータ管理を処理します。
    """

    fused_worker_attr_name = "fused_worker_dict"

    def __new__(cls, *args, **kwargs):
        """環境設定に基づいて適切な初期化を行う新しい Worker インスタンスを作成。"""
        instance = super().__new__(cls)

        disable_worker_init = int(os.environ.get("DISABLE_WORKER_INIT", 0))
        if disable_worker_init:
            return instance

        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        if None not in [rank, worker_group_prefix] and "ActorClass(" not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        return instance

    def _configure_before_init(self, register_center_name: str, rank: int):
        """初期化前にワーカー設定を構成。

        Args:
            register_center_name (str):
                ワーカー調整用の register center Ray actor の名前
            rank (int):
                分散設定におけるワーカーのランク
        """
        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"

        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }

            if os.getenv("WG_BACKEND", None) == "ray":
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center

                self.register_center = create_worker_group_register_center(
                    name=register_center_name, info=rank_zero_info
                )

            os.environ.update(rank_zero_info)
        else:
            self.register_center = ray.get_actor(register_center_name)

        ray.get(self.register_center.set_worker_info.remote(rank, ray.get_runtime_context().get_node_id()))

    @classmethod
    def env_keys(cls):
        """Worker の設定に使用される環境変数のキー。"""
        return [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            get_visible_devices_keyword().upper(),
        ]

    def __init__(self, cuda_visible_devices=None) -> None:
        """環境設定とデバイス設定でワーカーを初期化。

        Args:
            cuda_visible_devices (str, optional):
                CUDA visible devices 設定。デフォルトは None。
        """
        import os

        self._setup_env_cuda_visible_devices()

        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            "_world_size": world_size,
            "_rank": rank,
            "_local_world_size": local_world_size,
            "_local_rank": local_rank,
            "_master_addr": master_addr,
            "_master_port": master_port,
        }
        if cuda_visible_devices is not None:
            store[f"_{get_visible_devices_keyword()}".lower()] = cuda_visible_devices

        self._configure_with_store(store=store)

        self.fused_worker_dict = {}

    def get_fused_worker_by_name(self, worker_name: str):
        """名前によって fused worker を取得。

        Args:
            worker_name (str):
                取得するワーカーの名前
        """
        return self.fused_worker_dict.get(worker_name, None)

    def _setup_env_cuda_visible_devices(self):
        from verl.utils.ray_utils import ray_noset_visible_devices

        is_ray_noset_visible_devices = ray_noset_visible_devices()

        # Prevent use of clashing `{CUDA/HIP/ROCR}_VISIBLE_DEVICES``
        rocr_val = os.environ.get("ROCR_VISIBLE_DEVICES", None)
        hip_val = os.environ.get("HIP_VISIBLE_DEVICES", None)
        cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if hip_val:
            # Switch the use of HIP_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES for consistency.
            # Make sure that the HIP_VISIBLE_DEVICES is set to the same value as CUDA_VISIBLE_DEVICES
            # at this point.
            val = os.environ.pop("HIP_VISIBLE_DEVICES")
            hip_val = None
            if cuda_val:
                assert val == cuda_val, (
                    f"Please use the same HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, inconsistant values "
                    f"found: {val} and {cuda_val}."
                )
            else:
                cuda_val = val
                os.environ["CUDA_VISIBLE_DEVICES"] = val
                # os.environ["HIP_VISIBLE_DEVICES"] = val

        if rocr_val:
            # You must take care if both HIP/CUDA and ROCR env vars are set as they have
            # different meanings. Both env vars accept either a list of ints or a
            # list of UUIDs. The ROCR env var is processed first which then reduces
            # the number of GPUs that HIP can select from.
            # https://github.com/pytorch/pytorch/pull/144026
            # To avoid the complexity of this, we simply gives out error if both are set
            # (Also to keep consistency with ray's practice with 2.45.0).
            # Otherwise, we will set ROCR_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES
            # and remove ROCR_VISIBLE_DEVICES.
            if cuda_val:
                raise ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")

            cuda_val = os.environ.pop("ROCR_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val
            rocr_val = None

        if is_ray_noset_visible_devices:
            # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
            # environment variable for each actor, unless
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set,
            # so we need to set local rank when the flag is set.
            local_rank = os.environ.get("RAY_LOCAL_RANK")
            os.environ["LOCAL_RANK"] = local_rank
            get_torch_device().set_device(int(local_rank))

    def _configure_with_store(self, store: dict):
        """
        This function should only be called inside by WorkerGroup
        """
        store_env_dict = {f"_{key.lower()}": store.get(f"_{key.lower()}", None) for key in type(self).env_keys()}
        self.__dict__.update(store_env_dict)  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        for key in type(self).env_keys():
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                # print(f"set {key} to {val}")
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = (
            str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""
        )

    def get_master_addr_port(self):
        """Get the master address and port for distributed communication."""
        return self._master_addr, self._master_port

    def get_cuda_visible_devices(self):
        """Get the CUDA visible devices configuration."""
        import os

        visible_devices = os.environ.get(get_visible_devices_keyword().upper(), "not set")
        return visible_devices

    @property
    def world_size(self):
        """Get the total number of workers in the distributed setup."""
        return self._world_size

    @property
    def rank(self):
        """Get the rank of this worker in the distributed setup."""
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        """Execute a function with function generator dispatch mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        """Execute a function in rank zero execution mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        result = func(*args, **kwargs)
        return result
