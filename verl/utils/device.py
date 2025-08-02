# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """NPU の利用可能性をチェックする"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_visible_devices_keyword() -> str:
    """可視デバイスキーワード名を取得する関数。
    Returns:
        'CUDA_VISIBLE_DEVICES' または `ASCEND_RT_VISIBLE_DEVICES`
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """現在のマシンに基づいて torch.device を取得する関数。
    現在は CPU、CUDA、NPU のみをサポートしています。
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """デバイスタイプ文字列に基づいて対応する torch 属性を返す。
    Returns:
        module: 対応する torch デバイス名前空間、見つからない場合は torch.cuda
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"デバイス名前空間 '{device_name}' が torch で見つかりません。torch.cuda の読み込みを試行します。")
        return torch.cuda


def get_device_id() -> int:
    """デバイスタイプに基づいて現在のデバイス ID を返す。
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """デバイスタイプに基づいて nccl バックエンドタイプを返す。
    Returns:
        nccl バックエンドタイプ文字列
    """
    if is_cuda_available:
        return "nccl"
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(f"デバイスタイプ {get_device_name()} で利用可能な nccl バックエンドが見つかりません。")
