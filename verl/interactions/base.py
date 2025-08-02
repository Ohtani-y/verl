# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")  # より汎用的なエージェントのデフォルトロール名

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """ツールインスタンスを作成します。

        Args:
            instance_id: ツールのインスタンスID。

        Returns:
            ツールのインスタンスID。
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:  # より明確なレスポンス生成メソッド
        """
        現在のインタラクションターンのレスポンスを生成します。
        以下を含むタプルを返します：
        - should_terminate_sequence (bool): インタラクションシーケンスを終了すべき場合はTrue。
        - response_content (str): レスポンスのテキスト内容。
        - current_turn_score (float): この特定のターン/レスポンスのスコア。
        - additional_data (dict): 追加情報やメタデータ。
        """
        should_terminate_sequence: bool = False  # Trueの場合、ロールアウトを終了
        response_content: str = "Your current result seems acceptable."
        current_turn_score: float = 0.8
        additional_data: dict[str, Any] = {}
        return should_terminate_sequence, response_content, current_turn_score, additional_data

    async def calculate_score(self) -> float:  # より明確なスコア計算メソッド
        """
        インタラクションのスコアを計算します。
        部分的な露出やコンテキスト内タスク切り替えなどの側面を考慮する可能性があります。
        ターンレベルで呼び出されるべきです。
        """
        score = 0.0
        return score

    async def finalize_interaction(self) -> None:  # より明確なインタラクション終了とリソース解放メソッド
        """
        インタラクションセッションを終了し、関連する状態やリソースを解放します。
        シミュレート：状態の解放
        """
        pass
