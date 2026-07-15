# SPDX-License-Identifier: LGPL-3.0-only

import tempfile
import unittest
from pathlib import Path

from litellm.litellm_core_utils.token_counter import token_counter

from rotator_library.anthropic_compat.streaming_fast import (
    MiniMaxTextToolCallParser,
    anthropic_streaming_wrapper,
)
from rotator_library.anthropic_compat.translator import openai_to_anthropic_response
from rotator_library.context_compactor import CompactionConfig, ContextCompactor
from rotator_library.token_calculator import (
    MIN_MAX_TOKENS,
    calculate_max_tokens,
    count_input_tokens,
    count_input_tokens_result,
    estimate_input_tokens,
)
from rotator_library.usage_manager import UsageManager
from rotator_library.utils.json_utils import STREAM_DONE, json_loads


def _parse_sse_events(chunks):
    events = []
    event_name = None
    data_lines = []
    for line in "".join(chunks).splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
        elif not line and data_lines:
            events.append((event_name, json_loads("\n".join(data_lines))))
            event_name = None
            data_lines = []
    return events


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


class WindowsPlatformSeedTests(unittest.TestCase):
    def test_windows_platform_uname_seed_is_idempotent(self):
        import platform
        import sys

        from rotator_library.client import bootstrap

        if sys.platform != "win32":
            bootstrap._seed_windows_platform_uname()
            return

        original_cache = getattr(platform, "_uname_cache", None)
        try:
            platform._uname_cache = None

            bootstrap._seed_windows_platform_uname()
            seeded_cache = getattr(platform, "_uname_cache", None)

            self.assertIsNotNone(seeded_cache)
            self.assertEqual(seeded_cache.system, "Windows")

            bootstrap._seed_windows_platform_uname()
            self.assertIs(getattr(platform, "_uname_cache", None), seeded_cache)
        finally:
            platform._uname_cache = original_cache


class TokenEstimateTests(unittest.TestCase):
    def test_estimate_is_reasonably_close_for_common_payloads(self):
        samples = [
            "Plain English prose with punctuation, short words, and longer sentences.",
            "def handle(value):\n    return {'id': value, 'ok': value is not None}\n",
            '{"id":"00123","enabled":"true","items":[1,2,3],"nested":{"x":null}}',
            "Unicode text: Privet mir, konnichiwa sekai, emoji 😀🚀",
        ]

        for sample in samples:
            messages = [{"role": "user", "content": sample}]
            estimated, _bytes = estimate_input_tokens(messages=messages)
            exact = token_counter(model="gpt-4o-mini", messages=messages)
            self.assertGreaterEqual(estimated, exact, sample)
            self.assertLessEqual(estimated, exact * 6 + 64, sample)

    def test_large_payload_uses_fast_estimate_path(self):
        import rotator_library.token_calculator as token_calculator

        original_threshold_bytes = token_calculator.EXACT_TOKEN_COUNTER_MAX_BYTES
        original_threshold_chars = token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS
        token_calculator.EXACT_TOKEN_COUNTER_MAX_BYTES = 1
        token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS = 1
        try:
            messages = [{"role": "user", "content": "abc123_{}[] Privet 😀"}]
            estimated, _bytes = estimate_input_tokens(messages=messages)
            result = count_input_tokens_result(messages, "gpt-4o-mini")

            self.assertFalse(result.exact)
            self.assertEqual(result.count, estimated)
            self.assertEqual(count_input_tokens(messages, "gpt-4o-mini"), estimated)
        finally:
            token_calculator.EXACT_TOKEN_COUNTER_MAX_BYTES = original_threshold_bytes
            token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS = original_threshold_chars

    def test_estimated_overflow_does_not_hard_reject_request(self):
        import rotator_library.token_calculator as token_calculator

        original_threshold_bytes = token_calculator.EXACT_TOKEN_COUNTER_MAX_BYTES
        original_threshold_chars = token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS
        original_timeout = token_calculator.EXACT_TOKEN_COUNTER_TIMEOUT_SECONDS
        token_calculator.EXACT_TOKEN_COUNTER_MAX_BYTES = 1
        token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS = 1
        token_calculator.EXACT_TOKEN_COUNTER_TIMEOUT_SECONDS = 0.0
        try:
            messages = [{"role": "user", "content": "a" * 20000}]
            calculated, reason = calculate_max_tokens(
                model="gpt-4",
                messages=messages,
                safety_buffer=0,
            )

            self.assertEqual(calculated, MIN_MAX_TOKENS)
            self.assertIn("estimated_input_maybe_exceeds_context", reason)
        finally:
            token_calculator.EXACT_TOKEN_COUNTER_MAX_BYTES = original_threshold_bytes
            token_calculator.EXACT_TOKEN_COUNTER_MAX_CHARS = original_threshold_chars
            token_calculator.EXACT_TOKEN_COUNTER_TIMEOUT_SECONDS = original_timeout


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

    def test_parser_coerces_arguments_by_schema(self):
        parser = MiniMaxTextToolCallParser(
            enabled=True,
            tool_schemas={
                "Read": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "limit": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                        "items": {"type": "array"},
                    },
                }
            },
        )
        events = parser.feed(
            (
                '<tool_call><invoke name="Read">'
                "<id>00123</id>"
                "<limit>25</limit>"
                "<enabled>true</enabled>"
                "<items>[1,2]</items>"
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
                "limit": 25,
                "enabled": True,
                "items": [1, 2],
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

    def test_non_streaming_schema_coercion_for_recovered_tool_call(self):
        response = openai_to_anthropic_response(
            {
                "id": "chatcmpl_test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '<tool_call><invoke name="Read">'
                                "<id>00123</id>"
                                "<limit>25</limit>"
                                "<enabled>true</enabled>"
                                "<items>[1,2]</items>"
                                "</invoke></tool_call>"
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
            tool_schemas={
                "Read": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "limit": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                        "items": {"type": "array"},
                    },
                }
            },
        )

        self.assertEqual(
            response["content"][0]["input"],
            {
                "id": "00123",
                "limit": 25,
                "enabled": True,
                "items": [1, 2],
            },
        )

    def test_non_streaming_skips_recovered_call_when_structured_call_exists(self):
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
                            "tool_calls": [
                                {
                                    "id": "call_native",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": '{"file_path":"e:\\\\repo\\\\note.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
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

        tool_blocks = [block for block in response["content"] if block["type"] == "tool_use"]
        text = "".join(
            block["text"] for block in response["content"] if block["type"] == "text"
        )
        self.assertEqual(len(tool_blocks), 1)
        self.assertNotIn("<tool_call>", text)
        self.assertEqual(tool_blocks[0]["id"], "call_native")


class MinimaxStreamingRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovered_and_native_tool_call_indexes_do_not_collide(self):
        async def openai_stream():
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": (
                                '<tool_call><invoke name="Read">'
                                "<file_path>e:\\repo\\note.md</file_path>"
                                "</invoke></tool_call>"
                            )
                        },
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_native",
                                    "type": "function",
                                    "function": {
                                        "name": "Search",
                                        "arguments": '{"query":"note"}',
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
            yield STREAM_DONE

        chunks = [
            event
            async for event in anthropic_streaming_wrapper(
                openai_stream=openai_stream(),
                original_model="minimax/MiniMax-M3",
                request_id="msg_test",
            )
        ]
        events = _parse_sse_events(chunks)

        tool_starts = [
            payload["content_block"]["name"]
            for _event_name, payload in events
            if payload.get("type") == "content_block_start"
            and payload["content_block"]["type"] == "tool_use"
        ]
        deltas = [
            json_loads(payload["delta"]["partial_json"])
            for _event_name, payload in events
            if payload.get("type") == "content_block_delta"
            and payload["delta"]["type"] == "input_json_delta"
        ]

        self.assertEqual(tool_starts, ["Read", "Search"])
        self.assertEqual(deltas, [{"file_path": "e:\\repo\\note.md"}, {"query": "note"}])


if __name__ == "__main__":
    unittest.main()
