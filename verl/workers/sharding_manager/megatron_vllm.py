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
このファイルには、アクターの重みを推論エンジンと共有する Megatron スタイルの Hybrid Engine が含まれています。
"""

import inspect
import logging
import os

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig
from torch import nn

from verl import DataProto
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu, per_tensor_generator
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.profiler.performance import simple_timer
from verl.utils.torch_functional import check_device_is_available

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- トレーニング中は、現在の pp ステージのみがパラメータを保持
- 推論前に、現在の pp ランクのパラメータを他の全ての pp ランクにブロードキャスト
   （全ての pp ランクが全パラメータを保持）
- パラメータを推論エンジンにバインド
- tp で推論を実行。pp は追加の dp として扱われる
- 推論後、この pp ランクに属さない全パラメータを解放
"""


class MegatronVLLMShardingManager(BaseShardingManager):
    """Megatron-LM トレーニングと vLLM 推論を橋渡しするシャーディングマネージャー。

    このクラスは以下の間でのパラメータシャーディングと通信を処理します：
    - Megatron-LM のテンソル/エキスパート並列トレーニング設定
    - vLLM のテンソル並列推論設定

    主な責務：
    - トレーニングと推論設定間でのパラメータブロードキャストを管理
    - Megatron と HuggingFace 形式間での重み変換を処理
    - トレーニングと推論フェーズ間でのメモリ管理を調整
    - 異なる並列グループ間でのランダム状態の一貫性を維持

    Args:
        actor_module (nn.ModuleList): トレーニング中の Megatron-LM モデル
        inference_engine (LLM): vLLM 推論エンジン
        model_config: アクターモデルの設定
        transformer_config: モデルの Transformer 固有設定
        rollout_config: ロールアウトの設定
        layer_name_mapping: Megatron と HF レイヤー名のマッピング
        weight_converter (McoreToHFWeightConverterBase): 形式間で重みを変換
        device_mesh: 並列操作用のデバイスメッシュ
        offload_param (bool): 未使用時にパラメータをオフロードするかどうか
    """

    @check_device_is_available()
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: LLM,
        model_config: DictConfig,
        transformer_config,
        rollout_config: DictConfig,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        device_mesh,
        offload_param: bool = True,
        bridge=None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.offload_param = offload_param

        self.model_runner = (
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            if self.inference_engine
            else None
        )

        self.model_config = model_config
        self.transformer_config = transformer_config
        self.rollout_config = rollout_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.bridge = bridge
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        self.device_mesh = device_mesh
        self.infer_tp_size = self.device_mesh["infer_tp"].size()
        self.infer_tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        self.train_tp_size = mpu.get_tensor_model_parallel_world_size()
        self.train_tp_rank = mpu.get_tensor_model_parallel_rank()
        self.train_tp_group = mpu.get_tensor_model_parallel_group()
        self.train_ep_size = mpu.get_expert_model_parallel_world_size()
        self.train_ep_rank = mpu.get_expert_model_parallel_rank()
        self.train_ep_group = mpu.get_expert_model_parallel_group()
        self.train_etp_size = mpu.get_expert_tensor_parallel_world_size()
        self.train_etp_rank = mpu.get_expert_tensor_parallel_rank()
        self.train_etp_group = mpu.get_expert_tensor_parallel_group()
        self.need_tp_reshard = self.train_tp_size != self.infer_tp_size
        self.train_tp_larger = self.train_tp_size > self.infer_tp_size

        self.torch_random_states = get_torch_device().get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            aggressive_empty_cache(force_sync=True)

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_megatron_model_to_gpu(self.actor_module, load_grad=False)

            if self.rollout_config.free_cache_engine:
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["weights"])
                else:
                    self.inference_engine.wake_up()
            if self.bridge is not None:
                per_tensor_param = self.bridge.export_weights(self.actor_module)
            else:
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
            model = self.model_runner.model
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(model)
            loaded_params = model.load_weights(per_tensor_param)
            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            logger.info(info)

            if self.offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            aggressive_empty_cache(force_sync=True)

            if (
                self.rollout_config.free_cache_engine
                and "tags" in inspect.signature(self.inference_engine.wake_up).parameters
            ):
                self.inference_engine.wake_up(tags=["kv_cache"])

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self.rollout_config.free_cache_engine:
            self.inference_engine.sleep(level=1)
        for model in self.actor_module:
            model.train()

        aggressive_empty_cache(force_sync=True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.infer_tp_rank]
