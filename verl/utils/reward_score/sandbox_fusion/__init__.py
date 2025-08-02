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
import json
import logging
import traceback

from .utils import check_correctness

"""
Sandbox Fusion (https://github.com/bytedance/SandboxFusion) を使用してコードの正確性を検証します。
sandbox_fusion サービスを自分でデプロイするか、
パブリッククラウドが提供する FaaS サービス（例：volcengine.com）を使用できます。
"""
logger = logging.getLogger(__name__)


def compute_score(
    sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_cases, continuous=False, timeout=10
):
    """
    リモート sandbox API を使用してコードスコアを計算します。

    Args:
        sandbox_fusion_url: sandbox_fusion サービスの URL、例："https://<your service endpoint>/run_code"

        completion: コードを含む完了文字列。
        test_cases: "inputs" と "outputs" を含む JSON 文字列または辞書。
        continuous: 連続スコアを計算するかどうか（最初の N 個のテストケースに基づく）。
        timeout: 各テストケースのタイムアウト。

    Returns:
        タプル (score, metadata_list)。
        score: 浮動小数点スコア（0.0 から 1.0）。
        metadata_list: 各テストケースの実行メタデータを含むリスト。
    """
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # 言語名の簡単なチェック
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if test_cases is not None and "assert_case" in test_cases and isinstance(test_cases.get("assert_case"), list):
            assert_cases = test_cases.get("assert_case")
            test_cases.setdefault("inputs", ["" for _ in assert_cases])
            test_cases.setdefault("outputs", [None for _ in assert_cases])
        elif not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            logger.error("Invalid test_cases structure.")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        # results_list には True、False、またはエラーコード（-1、-2、-3 など）が含まれます
        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )

        if not res_list:  # 結果がない場合（例：無効な入力）
            return 0.0, metadata_list

        if continuous:
            num_to_consider = min(len(res_list), 10)
            if num_to_consider == 0:
                score = 0.0
            else:
                passed_count = sum(1 for r in res_list[:num_to_consider] if r is True)
                score = passed_count / num_to_consider
            # Return all metadata, even if score is based on the first N
            final_metadata = metadata_list
        else:
            # Calculate pass rate for all test cases
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            score = passed_count / total_cases if total_cases > 0 else 0.0
            final_metadata = metadata_list

    except Exception as e:
        logger.error(f"Error during compute_score: {e}")
        traceback.print_exc()
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    # Ensure float and list are returned
    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]
