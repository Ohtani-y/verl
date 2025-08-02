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
WorkerGroup クラス
"""

import logging
import signal
import threading
import time
from typing import Any, Callable

from .decorator import MAGIC_ATTR, Dispatch, get_predefined_dispatch_fn, get_predefined_execute_fn


class ResourcePool:
    """
    複数ノード間でリソースプールを管理し、プロセス数と GPU 割り当てを追跡します。
    このクラスは、プール内のすべてのノードにわたって world size、local world size、
    local rank を計算するメソッドを提供します。
    """

    def __init__(self, process_on_nodes=None, max_colocate_count: int = 10, n_gpus_per_node=8) -> None:
        """ノードプロセスと GPU 設定で ResourcePool を初期化します。

        Args:
            process_on_nodes (List[int], optional): ノードごとのプロセス数のリスト。デフォルトは空のリスト。
            max_colocate_count (int, optional): 同一場所に配置できるプロセスの最大数。デフォルトは 10。
            n_gpus_per_node (int, optional): ノードごとに利用可能な GPU 数。デフォルトは 8。
        """
        if process_on_nodes is None:
            process_on_nodes = []
        self._store = process_on_nodes
        self.max_colocate_count = max_colocate_count
        self.n_gpus_per_node = n_gpus_per_node  # これは将来の Huawei GPU（ノードあたり 16 GPU を含む）のために残されています

    def add_node(self, process_count):
        self._store.append(process_count)

    @property
    def world_size(self):
        """プール内のすべてのノードにわたるプロセスの総数。"""
        return sum(self._store)

    def __call__(self) -> Any:
        return self._store

    @property
    def store(self):
        return self._store

    def local_world_size_list(self) -> list[int]:
        """各プロセスがその local world size を持つフラットなリストを返します。"""
        nested_local_world_size_list = [
            [local_world_size for _ in range(local_world_size)] for local_world_size in self._store
        ]
        return [item for row in nested_local_world_size_list for item in row]

    def local_rank_list(self) -> list[int]:
        """すべてのノードにわたるすべてのプロセスの local rank のフラットなリストを返します。"""
        nested_local_rank_list = [[i for i in range(local_world_size)] for local_world_size in self._store]
        return [item for row in nested_local_rank_list for item in row]


class ClassWithInitArgs:
    """
    遅延インスタンス化のためにコンストラクタ引数を保存するラッパークラス。
    このクラスは、実際の構築が異なる時間や場所で行われる必要がある
    リモートクラスのインスタンス化に特に有用です。
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        """Initialize the ClassWithInitArgs instance.

        Args:
            cls: The class to be instantiated later
            *args: Positional arguments for the class constructor
            **kwargs: Keyword arguments for the class constructor
        """
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

        self.fused_worker_used = False

    def __call__(self) -> Any:
        """Instantiate the stored class with the stored arguments."""
        return self.cls(*self.args, **self.kwargs)


def check_workers_alive(workers: list, is_alive: Callable, gap_time: float = 1) -> None:
    """Continuously monitors worker processes and raises SIGABRT if any worker dies.

    Args:
        workers (List):
            List of worker objects to monitor
        is_alive (Callable):
            Function to check if a worker is alive
        gap_time (float):
            Time interval between checks
    """
    import time

    while True:
        for worker in workers:
            if not is_alive(worker):
                logging.warning(f"worker {worker} is not alive sending signal to main thread")
                signal.raise_signal(signal.SIGABRT)
        time.sleep(gap_time)


class WorkerGroup:
    """
    Base class for managing a group of workers in a distributed system.
    The class provides methods for worker management, aliveness checking, and method binding.
    """

    fused_worker_execute_fn_name = "_fuw_execute"

    def __init__(self, resource_pool: ResourcePool, **kwargs) -> None:
        self._is_init_with_detached_workers = resource_pool is None

        self.fused_worker_used = False

        if resource_pool is not None:
            # handle the case when WorkGroup is attached to an existing one
            self._procecss_dispatch_config = resource_pool()
        else:
            self._procecss_dispatch_config = None

        self._workers = []
        self._worker_names = []

        self._master_addr = None
        self._master_port = None

        self._checker_thread: threading.Thread = None

    def _is_worker_alive(self, worker):
        """Check if a worker is alive. Must be implemented by derived classes."""
        raise NotImplementedError("WorkerGroup._is_worker_alive called, should be implemented in derived class.")

    def _block_until_all_workers_alive(self) -> None:
        """Blocks until all workers in the group are alive."""
        while True:
            all_state = [self._is_worker_alive(worker) for worker in self._workers]
            if False in all_state:
                time.sleep(1)
            else:
                break

    def start_worker_aliveness_check(self, every_n_seconds=1) -> None:
        """Starts a background thread to monitor worker aliveness.

        Args:
            every_n_seconds (int): Interval between aliveness checks
        """
        # before starting checking worker aliveness, make sure all workers are already alive
        self._block_until_all_workers_alive()

        self._checker_thread = threading.Thread(
            target=check_workers_alive, args=(self._workers, self._is_worker_alive, every_n_seconds)
        )
        self._checker_thread.start()

    @property
    def world_size(self):
        """Number of workers in the group."""
        return len(self._workers)

    def _bind_worker_method(self, user_defined_cls, func_generator):
        """Binds worker methods to the WorkerGroup based on registered attributes.

        Args:
            user_defined_cls (type): The class containing methods to bind
            func_generator (Callable): Function that generates the bound method

        Returns:
            List[str]: List of method names that were successfully bound
        """
        method_names = []
        for method_name in dir(user_defined_cls):
            try:
                method = getattr(user_defined_cls, method_name)
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                # if it is a property, it will fail because Class doesn't have instance property
                continue

            if hasattr(method, MAGIC_ATTR):
                # this method is decorated by register
                attribute = getattr(method, MAGIC_ATTR)
                assert isinstance(attribute, dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]
                blocking = attribute["blocking"]

                # get dispatch fn
                if isinstance(dispatch_mode, Dispatch):
                    # get default dispatch fn
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                # get execute_fn_name
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]

                # get execute_fn from string
                try:
                    execute_fn = getattr(self, wg_execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception:
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    raise

                # bind a new method to the RayWorkerGroup
                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )

                try:
                    setattr(self, method_name, func)
                    method_names.append(method_name)
                except Exception as e:
                    raise ValueError(f"Fail to set method_name {method_name}") from e

        return method_names
