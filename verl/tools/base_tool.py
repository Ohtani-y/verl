# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import json
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema


class BaseTool:
    """ツールのベースクラス。

    ツールは以下のメソッドをサポートする必要があります：

    - `get_openai_tool_schema`: OpenAI 形式でツールスキーマを返す。
    - `create`: 軌跡用のツールインスタンスを作成する。
    - `execute`: ツールを実行する。
    - `calc_reward`: ツール状態に関する報酬を計算する。
    - `release`: ツールインスタンスを解放する。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name
        print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """ツールインスタンスを作成する。

        Args:
            instance_id: ツールのインスタンス ID。

        Returns:
            ツールのインスタンス ID。
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """ツールを実行する。

        Args:
            instance_id: ツールのインスタンス ID。
            parameters: ツールのパラメータの JSON 文字列。

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: ツールのレスポンス文字列。
            tool_reward_score: ツールのステップ報酬スコア。
            tool_metrics: ツールのメトリクス。
        """
        return "Updated the tool state.", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """ツールの報酬を計算する。

        Args:
            instance_id: ツールのインスタンス ID。

        Returns:
            ツールの報酬。
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """ツールインスタンスを解放する。

        Args:
            instance_id: ツールのインスタンス ID。
        """
        pass
