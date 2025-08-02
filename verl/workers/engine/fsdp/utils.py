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
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name


def create_device_mesh(world_size, fsdp_size):
    """
    world_size と FSDP サイズに基づいて分散トレーニング用のデバイスメッシュを作成します。

    Args:
        world_size (int): 分散トレーニング設定における総プロセス数
        fsdp_size (int): Fully Sharded Data Parallel (FSDP) グループのサイズ

    Returns:
        torch.distributed.device_mesh.DeviceMesh: 初期化されたデバイスメッシュ
    """
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    """
    デバイスメッシュの次元数に基づいて適切なシャーディング戦略を決定します。

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): 分散トレーニングに使用されるデバイスメッシュ

    Returns:
        torch.distributed.fsdp.ShardingStrategy: FSDP で使用されるシャーディング戦略

    Raises:
        NotImplementedError: デバイスメッシュの次元数が1でも2でもない場合
    """
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy
