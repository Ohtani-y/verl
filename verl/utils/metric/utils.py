# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
メトリクスユーティリティ。
"""

from typing import Any

import numpy as np


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    メトリクスリストの辞書を平均、最大、または最小値を計算して削減します。
    削減操作はキー名によって決定されます：
    - キーに "max" が含まれる場合、np.max が使用されます
    - キーに "min" が含まれる場合、np.min が使用されます
    - それ以外の場合、np.mean が使用されます

    Args:
        metrics: メトリクス名をメトリクス値のリストにマッピングする辞書。

    Returns:
        同じキーを持つが、各リストが削減された値に置き換えられた辞書。

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics
