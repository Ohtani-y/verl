# Copyright 2025 Bytedance Ltd. and/or its affiliates

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

import collections
from dataclasses import FrozenInstanceError, dataclass, field, fields
from typing import Any


# BaseConfig クラスは collections.abc.Mapping を継承し、辞書のように動作できます
@dataclass
class BaseConfig(collections.abc.Mapping):
    """BaseConfig は dataclass 設定に辞書ライクなインターフェースを提供します。

    デフォルトでは、"_mutable_fields" で指定されない限り、設定内のすべてのフィールドは
    変更不可です。BaseConfig クラスは Mapping Abstract Base Class を実装します。
    これにより、このクラスのインスタンスを辞書のように使用できます。
    """

    _mutable_fields = {"extra"}
    extra: dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value):
        """属性の値を設定します。値を設定する前に属性が変更可能かチェックします。"""
        if name in self.__dict__ and name not in getattr(self, "_mutable_fields", set()):
            raise FrozenInstanceError(f"Field '{name}' is frozen and cannot be modified")
        super().__setattr__(name, value)

    def get(self, key: str, default: Any = None) -> Any:
        """指定されたキーに関連付けられた値を取得します。キーが存在しない場合、デフォルト値を返します。

        Args:
            key (str): 取得する属性名。
            default (Any, optional): 属性が存在しない場合に返す値。デフォルトは None。

        Returns:
            Any: 属性の値またはデフォルト値。
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        """クラスの [] 演算子を実装します。辞書項目のように属性にアクセスできます。

        Args:
            key (str): 取得する属性名。

        Returns:
            Any: 属性の値。

        Raises:
            AttributeError: 属性が存在しない場合。
            TypeError: キーの型が文字列でない場合。
        """
        return getattr(self, key)

    def __iter__(self):
        """イテレータプロトコルを実装します。インスタンスの属性名を反復処理できます。

        Yields:
            str: dataclass 内の各フィールドの名前。
        """
        for f in fields(self):
            yield f.name

    def __len__(self):
        """
        dataclass 内のフィールド数を返します。

        Returns:
            int: dataclass 内のフィールド数。
        """
        return len(fields(self))
