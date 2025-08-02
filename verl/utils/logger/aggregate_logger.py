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
Ray ロガーは異なるプロセスからのログ情報を受信します。
"""

import datetime
import logging
import numbers
import pprint

import torch


def concat_dict_to_str(dict: dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    """
    コンソールにメッセージをログ出力するローカルロガー。

    Args:
        print_to_console (bool): コンソールに出力するかどうか。
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    """
    メッセージをログ出力するすべてのデコレータの基底クラス。

    Args:
        role (str): ロガーの役割（名前）。
        logger (logging.Logger): ログ出力に使用するロガーインスタンス。
        level (int): ログレベル。
        rank (int): プロセスのランク。
        log_only_rank_0 (bool): True の場合、ランク 0 のみログ出力。
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """分散処理が初期化されている場合、ランク 0 でのみ出力。"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """ランク情報付きでメッセージを出力。
    この関数は `log_only_rank_0` が False またはランクが 0 の場合のみメッセージを出力します。

    Args:
        message (str): 出力するメッセージ。
        rank (int, optional): プロセスのランク。デフォルトは 0。
        log_only_rank_0 (bool, optional): True の場合、ランク 0 のみ出力。デフォルトは False。
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """ランク情報とタイムスタンプ付きでメッセージを出力。
    この関数は `log_only_rank_0` が False またはランクが 0 の場合のみメッセージを出力します。

    Args:
        message (str): 出力するメッセージ。
        rank (int, optional): プロセスのランク。デフォルトは 0。
        log_only_rank_0 (bool, optional): True の場合、ランク 0 のみ出力。デフォルトは False。
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
