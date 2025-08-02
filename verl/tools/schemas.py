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
from typing import Any, Literal

from pydantic import BaseModel


class OpenAIFunctionPropertySchema(BaseModel):
    """OpenAI 形式のパラメータのスキーマ。"""

    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """OpenAI 形式のパラメータのスキーマ。"""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    """OpenAI 形式の関数のスキーマ。"""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """OpenAI 形式のツールのスキーマ。"""

    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    """OpenAI 形式のツールの解析済みスキーマ。"""

    name: str
    arguments: str  # JSON 文字列


class OpenAIFunctionCallSchema(BaseModel):
    """OpenAI 形式のツールの解析済みスキーマ。"""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    """OpenAI 形式のツール呼び出し。"""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema
