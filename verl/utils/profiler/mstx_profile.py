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

# Inspired from https://gitee.com/ascend/MindSpeed-RL/blob/master/mindspeed_rl/utils/utils.py
import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch_npu
from omegaconf import DictConfig
from torch_npu.npu import mstx

from .profile import DistProfiler, ProfilerConfig


def mark_start_range(message: Optional[str] = None) -> None:
    """プロファイラーでマーク範囲を開始します。

    Args:
        message (str, optional):
            プロファイラーに表示されるメッセージ。デフォルトは None。
    """
    return mstx.range_start(message=message)


def mark_end_range(range_id: str) -> None:
    """プロファイラーでマーク範囲を終了します。

    Args:
        range_id (str):
            終了するマーク範囲の ID。
    """
    return mstx.range_end(range_id)


def mark_annotate(message: Optional[str] = None) -> Callable:
    """関数のライフサイクルに沿ってマーク範囲を注釈する関数デコレーター。

    Args:
        message (str, optional):
            プロファイラーに表示されるメッセージ。デフォルトは None。
    """

    def decorator(func):
        profile_message = message or func.__name__
        return mstx.mstx_range(profile_message)(func)

    return decorator


@contextmanager
def marked_timer(name: str, timing_raw: dict[str, float], *args: Any, **kwargs: Any) -> None:
    """MSTX マーカーを使用したタイミング測定のコンテキストマネージャー。

    このユーティリティ関数は、コンテキスト内のコードの実行時間を測定し、
    タイミング情報を蓄積し、プロファイリング用の MSTX マーカーを追加します。

    Args:
        name (str): このタイミング測定の名前/識別子。
        timing_raw (Dict[str, float]): タイミング情報を格納する辞書。

    Yields:
        None: コードブロックに制御を戻すコンテキストマネージャー。
    """
    if args:
        logging.warning(f"Args are not supported in mstx_profile, but received: {args}")
    if kwargs:
        logging.warning(f"Kwargs are not supported in mstx_profile, but received: {kwargs}")
    mark_range = mark_start_range(message=name)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


def get_npu_profiler(option: DictConfig, role: Optional[str] = None, profile_step: Optional[str] = None):
    """NPU プロファイラーオブジェクトを生成して返します。

    Args:
        option (DictConfig):
            NPU プロファイラーを制御するオプション。
        role (str, optional):
            現在のデータ収集の役割。デフォルトは None。
        profile_step(str, optional):
            現在のトレーニングステップ。デフォルトは None。
    """
    if option.level == "level_none":
        profile_level = torch_npu.profiler.ProfilerLevel.Level_none
    elif option.level == "level0":
        profile_level = torch_npu.profiler.ProfilerLevel.Level0
    elif option.level == "level1":
        profile_level = torch_npu.profiler.ProfilerLevel.Level1
    elif option.level == "level2":
        profile_level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(f"level only supports level0, 1, 2, and level_none, but gets {option.level}")

    profile_save_path = option.save_path
    if profile_step:
        profile_save_path = os.path.join(profile_save_path, profile_step)
    if role:
        profile_save_path = os.path.join(profile_save_path, role)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=profile_level,
        export_type=torch_npu.profiler.ExportType.Text,
        data_simplification=True,
        msprof_tx=True,
    )

    activites = []
    if option.with_npu:
        activites.append(torch_npu.profiler.ProfilerActivity.NPU)
    if option.with_cpu:
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

    prof = torch_npu.profiler.profile(
        with_modules=option.with_module,
        with_stack=option.with_stack,
        record_shapes=option.record_shapes,
        profile_memory=option.with_memory,
        activities=activites,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path, analyse_flag=option.analysis),
        experimental_config=experimental_config,
    )
    return prof


class NPUProfiler(DistProfiler):
    """
    NPU profiler. Initialized in a worker to control the NPU profiler.
    """

    _define_count = 0

    def __init__(self, rank: int, config: ProfilerConfig, **kwargs):
        """Initialize the NsightSystemsProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
        """
        if not config:
            config = ProfilerConfig(ranks=[])
        self.this_step: bool = False
        self.discrete: bool = config.discrete
        self.this_rank: bool = False
        self.profile_npu = None
        self.profile_option = kwargs.get("option", None)
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        role, profile_step = kwargs.get("role", None), kwargs.get("profile_step", None)
        profile_step = str(profile_step) if profile_step is not None else None
        if self.this_rank and self.profile_option is not None:
            self.this_step = True
            if not self.discrete and NPUProfiler._define_count == 0:
                self.profile_npu = get_npu_profiler(option=self.profile_option, role=role, profile_step=profile_step)
                self.profile_npu.start()
                NPUProfiler._define_count += 1

    def stop(self):
        if self.this_rank and self.profile_option is not None:
            self.this_step = False
            if not self.discrete and NPUProfiler._define_count == 1:
                self.profile_npu.step()
                self.profile_npu.stop()
                NPUProfiler._define_count -= 1

    @staticmethod
    def annotate(message: Optional[str] = None, role: Optional[str] = None, **kwargs) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with NPUProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                profile_name = message or func.__name__
                profile_this_role = True
                discrete_mode = self.profiler.discrete
                profile_enable = self.profiler.this_step and self.profile_option is not None

                if not profile_enable:
                    return func(self, *args, **kwargs)

                if profile_enable and role is not None:
                    target_roles = self.profile_option.get("roles", [])
                    profile_this_role = "all" in target_roles or role in target_roles

                if profile_enable:
                    if not discrete_mode:
                        mark_range = mark_start_range(message=profile_name)
                    else:
                        if profile_this_role:
                            profile_npu = get_npu_profiler(option=self.profile_option, role=role)
                            profile_npu.start()
                            mark_range = mark_start_range(message=profile_name)

                result = func(self, *args, **kwargs)

                if profile_enable:
                    if not discrete_mode:
                        mark_end_range(mark_range)
                    else:
                        if profile_this_role:
                            mark_end_range(mark_range)
                            profile_npu.step()
                            profile_npu.stop()

                return result

            return wrapper

        return decorator
