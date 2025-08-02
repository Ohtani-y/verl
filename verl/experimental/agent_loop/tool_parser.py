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
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod

import regex as re
from pydantic import BaseModel

from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FunctionCall(BaseModel):
    arguments: str
    """
    モデルによって JSON 形式で生成された、関数を呼び出すための引数。
    モデルが常に有効な JSON を生成するとは限らず、関数スキーマで定義されていない
    パラメータを幻覚する可能性があることに注意してください。関数を呼び出す前に
    コード内で引数を検証してください。
    """

    name: str
    """呼び出す関数の名前。"""


class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """レスポンスからツール呼び出しを抽出します。

        Args:
            responses_ids (List[int]): レスポンスの ID。

        Returns:
            Tuple[str, List[FunctionCall]]: コンテンツと抽出されたツール呼び出し。
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    """https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py から適応"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        content = self.tool_call_regex.sub("", text)

        return content, function_calls
