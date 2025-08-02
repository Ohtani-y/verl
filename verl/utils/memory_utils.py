# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import gc
import logging

from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)


def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3) -> None:
    """
    より積極的な GPU メモリクリーンアップ関数。PyTorch が予約しているが未割り当てのメモリを解放しようとします。

    Args:
        force_sync: デバイス同期を強制するかどうか
        max_retries: 最大再試行回数
    """
    device = get_torch_device()
    if not device.is_available():
        return

    for attempt in range(max_retries):
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        gc.collect()

        device.empty_cache()

        if force_sync:
            device.synchronize()

        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()

        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated

        logger.info(
            f"Memory cleanup attempt {attempt + 1}: Freed {reserved_freed / 1024**3:.2f} GB reserved, "
            f"{allocated_freed / 1024**3:.2f} GB allocated"
        )

        if reserved_freed < 1024**3:  # 1GB未満
            break


def reset_memory_stats() -> None:
    """GPU メモリ統計をリセット"""
    if get_torch_device().is_available():
        device = get_torch_device()
        device.reset_peak_memory_stats()
        device.reset_accumulated_memory_stats()


def get_memory_info() -> dict:
    """詳細な GPU メモリ情報を取得"""
    if not get_torch_device().is_available():
        return {}

    device = get_torch_device()
    device_id = device.current_device()

    return {
        "total_memory_gb": device.get_device_properties(device_id).total_memory / 1024**3,
        "reserved_memory_gb": device.memory_reserved() / 1024**3,
        "allocated_memory_gb": device.memory_allocated() / 1024**3,
        "cached_memory_gb": (device.memory_reserved() - device.memory_allocated()) / 1024**3,
        "max_memory_allocated_gb": device.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": device.max_memory_reserved() / 1024**3,
    }


def log_memory_usage(stage: str = "current") -> None:
    """GPU メモリ使用量をログ出力"""
    if not get_torch_device().is_available():
        return

    info = get_memory_info()
    logger.info(
        f"Memory usage [{stage}]: "
        f"Total: {info['total_memory_gb']:.2f} GB, "
        f"Allocated: {info['allocated_memory_gb']:.2f} GB, "
        f"Reserved: {info['reserved_memory_gb']:.2f} GB, "
        f"Cached: {info['cached_memory_gb']:.2f} GB"
    )


def optimize_memory_for_inference() -> None:
    """推論用に GPU メモリ使用量を最適化"""
    if not get_torch_device().is_available():
        return

    get_torch_device().set_per_process_memory_fraction(0.95)  # GPU メモリの95%を使用

    aggressive_empty_cache(force_sync=True)

    logger.info("推論用に GPU メモリ使用量を最適化しました")


def optimize_memory_for_training() -> None:
    """Optimize GPU memory usage for training"""
    if not get_torch_device().is_available():
        return

    # Set a moderate memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=False)

    logger.info("Optimized GPU memory usage for training")
