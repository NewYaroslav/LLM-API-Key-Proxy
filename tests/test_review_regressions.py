# SPDX-License-Identifier: LGPL-3.0-only

import tempfile
import unittest
from pathlib import Path

from litellm.litellm_core_utils.token_counter import token_counter

from rotator_library.anthropic_compat.streaming_fast import MiniMaxTextToolCallParser
from rotator_library.anthropic_compat.translator import openai_to_anthropic_response
from rotator_library.context_compactor import CompactionConfig, ContextCompactor
from rotator_library.token_calculator import count_input_tokens, estimate_input_tokens
from rotator_library.usage_manager import UsageManager
from rotator_library.utils.json_utils import json_loads


class UsageManagerCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_before_lazy_init_is_noop_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = UsageManager(file_path=Path(tmp) / "usage.json")

            await manager.close()
            await manager.close()

    async def test_close_with_batch_persistence_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = UsageManager(file_path=Path(tmp) / "usage.json")
            manager._use_batch_persistence = True
            await manager._lazy_init()

            await manager.close()
            await manager.close()

            self.assertIsNone(manager._batch_persistence)

    async def test_close_without_batch_persistence_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = UsageManager(file_path=Path(tmp) / "usage.json")
            manager._use_batch_persistence = False
            await manager._lazy_init()

            await manager.close()
            await manager.close()


class TokenEstimateTests(unittest.TestCase):
    def test_estimate_is_not_below_real_tokenizer_for_common_payloads(self):
        samples = [
            "Plain English prose with punctuation, short words, and longer sentences.",
            "def handle(value):\n    return {'id': value, 'ok': value is not None}\n",
            '{"id":"00123","enabled":"true","items":[1,2,3],"nested":{"x":null}}',
            "Unicode text: Привет мир, こんにちは世界, emoji 😀🚀",
        ]

        for sample in samples:
            messages = [{"role": "user", "content": sample}]
            estimated, _chars = estimate_input_tokens(messages=messages)
            exact = token_counter(model="gpt-4o-mini", messages=messages)
            self.assertGreaterEqual(estimated, exact, sample)

    def test_large_payload_uses_conservative_fast_path(self):
        import rotator_library.token_calculator as token_calculator

        original_threshold = token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS
        token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS = 1
        try:
            messages = [{"role": "user", "content": "abc123_{}[] Привет 😀"}]
            estimated, _chars = estimate_input_tokens(messages=messages)
            self.assertEqual(
                count_input_tokens(messages, "gpt-4o-mini"),
                estimated,
            )
        finally:
            token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS = original_threshold


class ContextCompactorFastPathTests(unittest.TestCase):
    def test_noop_compaction_returns_original_payload(self):
        payload = {"messages": [{"role": "user", "content": "small"}]}
        compactor = ContextCompactor(
            CompactionConfig(enabled=True),
            token_counter=lambda _messages, _model: 1,
        )

        result = compactor.compact(
            payload,
            context_window=100,
            model="test/model",
        )

        self.assertIs(result, payload)


class MinimaxToolCallRegressionTests(unittest.TestCase):
    def test_parser_preserves_argument_strings_without_schema(self):
        parser = MiniMaxTextToolCallParser(enabled=True)
        events = parser.feed(
            (
                '<tool_call><invoke name="Lookup">'
                "<id>00123</id>"
                "<enabled>true</enabled>"
                "<empty>null</empty>"
                "<big>9007199254740993</big>"
                "</invoke></tool_call>"
            ),
            final=True,
        )

        tool_call = next(value for field, value in events if field == "tool_call")
        arguments = json_loads(tool_call["function"]["arguments"])

        self.assertEqual(
            arguments,
            {
                "id": "00123",
                "enabled": "true",
                "empty": "null",
                "big": "9007199254740993",
            },
        )

    def test_non_streaming_recovered_tool_call_preserves_block_order(self):
        response = openai_to_anthropic_response(
            {
                "id": "chatcmpl_test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "before"
                                '<tool_call><invoke name="Read">'
                                "<file_path>e:\\repo\\note.md</file_path>"
                                "</invoke></tool_call>"
                                "after"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
            "minimax/MiniMax-M3",
        )

        self.assertEqual(
            [block["type"] for block in response["content"]],
            ["text", "tool_use", "text"],
        )
        self.assertEqual(response["content"][0]["text"], "before")
        self.assertEqual(response["content"][1]["name"], "Read")
        self.assertEqual(response["content"][2]["text"], "after")


if __name__ == "__main__":
    unittest.main()
