# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Optimized streaming wrapper for converting OpenAI streaming format to Anthropic streaming format.

This module provides a framework-agnostic streaming wrapper that converts
OpenAI SSE (Server-Sent Events) format to Anthropic's streaming format.

Performance optimizations:
- Uses orjson for fast JSON parsing (3-5x faster than stdlib json)
- Reuses parsed objects where possible
- Minimizes allocations in hot paths
- Pre-built templates for common events
"""

import asyncio
import html
import logging
import httpx
import re
from time import monotonic
import uuid
from typing import AsyncGenerator, Callable, Optional, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..transaction_logger import TransactionLogger

from ..utils.json_utils import json_dumps_str as json_dumps, json_loads, STREAM_DONE

logger = logging.getLogger("rotator_library.anthropic_compat")

_THINK_START = "<think>"
_THINK_END = "</think>"
_MINIMAX_SEPARATOR_LONG = "]<]minimax[>["
_MINIMAX_SEPARATOR = "<]minimax[>["
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_INVOKE_END = "</invoke>"
_INVOKE_START_RE = re.compile(r"<invoke\s+name=(['\"])(?P<name>.*?)\1\s*>", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(
    r"<(?P<name>[A-Za-z_][\w.-]*)>(?P<value>.*?)</(?P=name)>",
    re.DOTALL,
)


class ThinkTagParser:
    __slots__ = ("enabled", "in_thinking", "pending")

    """Split streamed ``<think>`` tags into reasoning/text segments."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.in_thinking = False
        self.pending = ""

    @staticmethod
    def _suffix_prefix_len(text: str, marker: str) -> int:
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0

    def feed(self, text: str, final: bool = False) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self.enabled:
            return [("content", text)]

        data = self.pending + text
        self.pending = ""
        parts: list[tuple[str, str]] = []
        pos = 0

        while pos < len(data):
            marker = _THINK_END if self.in_thinking else _THINK_START
            idx = data.find(marker, pos)
            if idx >= 0:
                if idx > pos:
                    parts.append((
                        "reasoning_content" if self.in_thinking else "content",
                        data[pos:idx],
                    ))
                self.in_thinking = not self.in_thinking
                pos = idx + len(marker)
                continue

            tail = data[pos:]
            if not final:
                hold_len = self._suffix_prefix_len(tail, marker)
                if hold_len:
                    emit_text = tail[:-hold_len]
                    if emit_text:
                        parts.append((
                            "reasoning_content" if self.in_thinking else "content",
                            emit_text,
                        ))
                    self.pending = tail[-hold_len:]
                    return parts

            parts.append((
                "reasoning_content" if self.in_thinking else "content",
                tail,
            ))
            return parts

        return parts

    def flush(self) -> list[tuple[str, str]]:
        if not self.pending:
            return []
        text = self.pending
        self.pending = ""
        return [("reasoning_content" if self.in_thinking else "content", text)]


def _should_parse_think_tags(model: str) -> bool:
    model_lower = (model or "").lower()
    return (
        model_lower.startswith("minimax")
        or "/minimax/" in model_lower
        or "minimax-" in model_lower
    )


def _should_parse_text_tool_calls(model: str) -> bool:
    return _should_parse_think_tags(model)


def _strip_minimax_separators(text: str) -> str:
    if not text:
        return text
    return text.replace(_MINIMAX_SEPARATOR_LONG, "").replace(_MINIMAX_SEPARATOR, "")


def _coerce_tool_value(value: str) -> Any:
    # Minimax text-form tool calls do not include the original JSON Schema, so
    # heuristic coercion can corrupt string IDs, leading zero codes, and string
    # flags such as "true" or "null". Preserve values as strings.
    return value


class MiniMaxTextToolCallParser:
    __slots__ = ("enabled", "pending", "in_tool_call", "_next_index")

    """Recover MiniMax text-form tool calls from streamed content."""

    _SAFE_TAIL_LEN = max(
        len(_MINIMAX_SEPARATOR_LONG),
        len(_TOOL_CALL_START),
        len(_TOOL_CALL_END),
        len("<invoke name=\""),
    ) - 1

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.pending = ""
        self.in_tool_call = False
        self._next_index = 0

    @staticmethod
    def _suffix_prefix_len(text: str) -> int:
        markers = (
            _MINIMAX_SEPARATOR_LONG,
            _MINIMAX_SEPARATOR,
            _TOOL_CALL_START,
            _TOOL_CALL_END,
            "<invoke",
        )
        max_len = min(len(text), max(len(marker) for marker in markers) - 1)
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in markers):
                return size
        return 0

    @staticmethod
    def _clean_text(text: str) -> str:
        return _strip_minimax_separators(text)

    def _strip_orphan_tool_call_end(self) -> bool:
        stripped = self.pending.lstrip()
        lowered = stripped.lower()
        if lowered.startswith(_TOOL_CALL_END):
            self.pending = stripped[len(_TOOL_CALL_END):]
            return True
        if lowered.startswith("]" + _TOOL_CALL_END):
            self.pending = stripped[len(_TOOL_CALL_END) + 1:]
            return True
        return False

    def _make_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool_call = {
            "index": self._next_index,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json_dumps(arguments),
            },
        }
        self._next_index += 1
        return tool_call

    def _parse_invoke(self, raw: str) -> Optional[dict[str, Any]]:
        match = _INVOKE_START_RE.search(raw)
        if not match:
            return None

        name = html.unescape(match.group("name").strip())
        if not name:
            return None

        end_idx = raw.lower().rfind(_INVOKE_END)
        if end_idx < match.end():
            return None

        body = raw[match.end():end_idx]
        arguments: dict[str, Any] = {}
        for param in _PARAM_RE.finditer(body):
            key = html.unescape(param.group("name"))
            value = html.unescape(self._clean_text(param.group("value")))
            arguments[key] = _coerce_tool_value(value)

        return self._make_tool_call(name, arguments)

    def feed(self, text: str, final: bool = False) -> list[tuple[str, Any]]:
        if not text and not final:
            return []
        if not self.enabled:
            return [("content", text)] if text else []

        self.pending += text
        parts: list[tuple[str, Any]] = []

        while self.pending:
            self.pending = self._clean_text(self.pending)
            lower = self.pending.lower()

            if not self.in_tool_call and self._strip_orphan_tool_call_end():
                continue

            if self.in_tool_call:
                tool_end_idx = lower.find(_TOOL_CALL_END)
                invoke_idx = lower.find("<invoke")

                if invoke_idx < 0:
                    if tool_end_idx >= 0:
                        self.pending = self.pending[tool_end_idx + len(_TOOL_CALL_END):]
                        self.in_tool_call = False
                        continue
                    if final:
                        self.pending = ""
                    break

                if tool_end_idx >= 0 and tool_end_idx < invoke_idx:
                    self.pending = self.pending[tool_end_idx + len(_TOOL_CALL_END):]
                    self.in_tool_call = False
                    continue

                if invoke_idx > 0:
                    self.pending = self.pending[invoke_idx:]
                    lower = self.pending.lower()

                invoke_end_idx = lower.find(_INVOKE_END)
                if invoke_end_idx < 0:
                    if final:
                        self.pending = ""
                    break

                raw_invoke = self.pending[:invoke_end_idx + len(_INVOKE_END)]
                if tool_call := self._parse_invoke(raw_invoke):
                    parts.append(("tool_call", tool_call))

                self.pending = self.pending[invoke_end_idx + len(_INVOKE_END):]
                lower = self.pending.lower()
                if lower.startswith(_TOOL_CALL_END):
                    self.pending = self.pending[len(_TOOL_CALL_END):]
                    self.in_tool_call = False
                continue

            tool_idx = lower.find(_TOOL_CALL_START)
            invoke_idx = lower.find("<invoke")
            candidates = [idx for idx in (tool_idx, invoke_idx) if idx >= 0]
            start_idx = min(candidates) if candidates else -1

            if start_idx < 0:
                if final:
                    emit_text = self.pending
                    self.pending = ""
                else:
                    hold_len = max(self._suffix_prefix_len(self.pending), self._SAFE_TAIL_LEN)
                    if len(self.pending) <= hold_len:
                        break
                    emit_text = self.pending[:-hold_len]
                    self.pending = self.pending[-hold_len:]
                if emit_text := self._clean_text(emit_text):
                    parts.append(("content", emit_text))
                break

            if start_idx > 0:
                emit_text = self.pending[:start_idx]
                if emit_text := self._clean_text(emit_text):
                    parts.append(("content", emit_text))
                self.pending = self.pending[start_idx:]
                lower = self.pending.lower()

            if lower.startswith(_TOOL_CALL_START):
                self.pending = self.pending[len(_TOOL_CALL_START):]
                self.in_tool_call = True
                continue

            invoke_end_idx = lower.find(_INVOKE_END)
            if invoke_end_idx < 0:
                if final:
                    self.pending = ""
                break

            raw_invoke = self.pending[:invoke_end_idx + len(_INVOKE_END)]
            if tool_call := self._parse_invoke(raw_invoke):
                parts.append(("tool_call", tool_call))
                self.pending = self.pending[invoke_end_idx + len(_INVOKE_END):]
            else:
                emit_text = self.pending[:invoke_end_idx + len(_INVOKE_END)]
                if emit_text := self._clean_text(emit_text):
                    parts.append(("content", emit_text))
                self.pending = self.pending[invoke_end_idx + len(_INVOKE_END):]

        return parts

    def flush(self) -> list[tuple[str, Any]]:
        return self.feed("", final=True)


class ChunkBatcher:
    __slots__ = ("buffer", "current_size", "max_size", "max_delay_ms", "last_flush", "_first_event")

    """
    Batch SSE events for improved throughput.

    Buffers small events and flushes when:
    - Buffer size exceeds max_size
    - Time since last flush exceeds max_delay_ms
    """

    def __init__(self, max_size: int = 16384, max_delay_ms: int = 1):
        self.buffer = []
        self.current_size = 0
        self.max_size = max_size
        self.max_delay_ms = max_delay_ms
        self.last_flush = monotonic()
        self._first_event = True  # Flush first event immediately for low TTFT

    def add(self, event: str) -> Optional[str]:
        """
        Add event to buffer.

        Returns:
            Flushed buffer if threshold reached, None otherwise
        """
        self.buffer.append(event)
        self.current_size += len(event)

        # First event flushes immediately to minimize time-to-first-token
        if self._first_event:
            self._first_event = False
            return self.flush()

        if self.current_size >= self.max_size:
            return self.flush()

        # Check time-based flush
        elapsed_ms = (monotonic() - self.last_flush) * 1000
        if elapsed_ms >= self.max_delay_ms:
            return self.flush()

        return None

    def flush(self) -> str:
        """Flush buffer and return concatenated events."""
        if not self.buffer:
            return ""

        result = "".join(self.buffer)
        self.buffer.clear()
        self.current_size = 0
        self.last_flush = monotonic()
        return result


# Pre-built event templates for common operations (reduces allocations)
def _make_message_start_event(request_id: str, model: str, input_tokens: int = 0, cached_tokens: int = 0, cache_creation_tokens: int = 0) -> str:
    """Build message_start event string."""
    usage = {
        "input_tokens": input_tokens - cached_tokens - cache_creation_tokens,
        "output_tokens": 0,
    }
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens
    if cache_creation_tokens > 0:
        usage["cache_creation_input_tokens"] = cache_creation_tokens

    message_start = {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage,
        },
    }
    return f"event: message_start\ndata: {json_dumps(message_start)}\n\n"


def _make_content_block_start_event(index: int, block_type: str, cache_control: Optional[dict] = None, **extra) -> str:
    """Build content_block_start event string."""
    block: dict[str, Any] = {"type": block_type}
    if block_type == "text":
        block["text"] = ""
    elif block_type == "thinking":
        block["thinking"] = ""
    elif block_type == "tool_use":
        block.update(extra)

    if cache_control:
        block["cache_control"] = cache_control

    event = {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    }
    return f"event: content_block_start\ndata: {json_dumps(event)}\n\n"


def _make_text_delta_event(index: int, text: str) -> str:
    """Build text_delta event string."""
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json_dumps(event)}\n\n"


def _make_thinking_delta_event(index: int, thinking: str) -> str:
    """Build thinking_delta event string."""
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "thinking_delta", "thinking": thinking},
    }
    return f"event: content_block_delta\ndata: {json_dumps(event)}\n\n"


def _make_input_json_delta_event(index: int, partial_json: str) -> str:
    """Build input_json_delta event string."""
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }
    return f"event: content_block_delta\ndata: {json_dumps(event)}\n\n"


def _make_content_block_stop_event(index: int) -> str:
    """Build content_block_stop event string."""
    return f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {index}}}\n\n'


def _make_message_delta_event(stop_reason: str, output_tokens: int = 0, cached_tokens: int = 0, cache_creation_tokens: int = 0) -> str:
    """Build message_delta event string."""
    usage = {"output_tokens": output_tokens}
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens
    if cache_creation_tokens > 0:
        usage["cache_creation_input_tokens"] = cache_creation_tokens

    event = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage,
    }
    return f"event: message_delta\ndata: {json_dumps(event)}\n\n"


def _make_message_stop_event() -> str:
    """Build message_stop event string."""
    return 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


_ANTHROPIC_ERROR_TYPE_MAP = {
    "authentication": "authentication_error",
    "forbidden": "permission_error",
    "invalid_request": "invalid_request_error",
    "context_window_exceeded": "invalid_request_error",
    "not_found": "not_found_error",
    "model_not_found": "not_found_error",
    "quota_exceeded": "rate_limit_error",
    "rate_limit": "rate_limit_error",
    "ip_rate_limit": "rate_limit_error",
    "server_error": "api_error",
    "api_connection": "api_error",
    "proxy_account_billing_error": "api_error",
    "proxy_busy": "api_error",
    "proxy_internal_error": "api_error",
}


def _make_error_event(error_details: Any) -> str:
    """Build an Anthropic SSE error event from an internal OpenAI-style error."""
    if isinstance(error_details, dict):
        message = str(error_details.get("message") or "Stream failed")
        raw_error_type = str(error_details.get("type") or "api_error")
    else:
        message = str(error_details or "Stream failed")
        raw_error_type = "api_error"

    error_type = _ANTHROPIC_ERROR_TYPE_MAP.get(raw_error_type, "api_error")
    error_event = {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
    return f"event: error\ndata: {json_dumps(error_event)}\n\n"


# Sentinel returned by _StreamTranslator._parse_raw_chunk when SSE [DONE] is received
_PARSED_DONE = object()


class _StreamTranslator:
    """Stateful OpenAI-to-Anthropic SSE stream translator.

    Encapsulates all per-request state and dispatches chunk events
    to specialised private handlers.
    """

    _HEARTBEAT_INTERVAL = 30  # seconds
    _DISCONNECT_CHECK_INTERVAL = 50  # chunks

    __slots__ = (
        "request_id", "original_model", "is_disconnected", "transaction_logger",
        "batcher", "_batch_add",
        "message_started", "content_block_started", "thinking_block_started",
        "current_block_index",
        "tool_calls_by_index", "tool_block_indices",
        "input_tokens", "output_tokens", "cached_tokens", "cache_creation_tokens",
        "_text_parts", "_thinking_parts",
        "last_event_time", "_chunk_count",
        "_cache_control_map", "_think_tag_parser", "_text_tool_parser",
    )

    def __init__(
        self,
        request_id: str,
        original_model: str,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]],
        transaction_logger: Optional["TransactionLogger"],
        precomputed_input_tokens: Optional[int],
        cache_control_map: Optional[dict] = None,
    ):
        self.request_id = request_id
        self.original_model = original_model
        self.is_disconnected = is_disconnected
        self.transaction_logger = transaction_logger

        self.batcher = ChunkBatcher(max_size=16384, max_delay_ms=1)
        self._batch_add = self.batcher.add

        self.message_started = False
        self.content_block_started = False
        self.thinking_block_started = False
        self.current_block_index = 0

        self.tool_calls_by_index: dict = {}
        self.tool_block_indices: dict = {}

        self.input_tokens = precomputed_input_tokens if precomputed_input_tokens is not None else 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.cache_creation_tokens = 0

        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []

        self.last_event_time = monotonic()
        self._chunk_count = 0
        self._cache_control_map = cache_control_map or {}
        self._think_tag_parser = ThinkTagParser(_should_parse_think_tags(original_model))
        self._text_tool_parser = MiniMaxTextToolCallParser(_should_parse_text_tool_calls(original_model))

    # ------------------------------------------------------------------
    # Chunk parsing
    # ------------------------------------------------------------------

    def _parse_raw_chunk(self, raw_chunk: Any) -> Any:
        """Parse a raw chunk from the upstream stream.

        Returns:
            dict -- successfully parsed chunk,
            _PARSED_DONE -- SSE ``[DONE]`` received (stream should finalize),
            None -- unparseable / irrelevant chunk (skip).
        """
        if isinstance(raw_chunk, dict):
            return raw_chunk
        if isinstance(raw_chunk, str):
            if not raw_chunk or not raw_chunk.startswith("data:"):
                return None
            data_content = raw_chunk[5:].strip()
            if data_content == "[DONE]":
                return _PARSED_DONE
            try:
                return json_loads(data_content)
            except (ValueError, KeyError, TypeError) as e:
                logger.debug("Failed to parse SSE data chunk: %s: %s", data_content, e, exc_info=True)
                return None
        return None

    # ------------------------------------------------------------------
    # Usage extraction
    # ------------------------------------------------------------------

    def _extract_usage(self, chunk: dict) -> None:
        """Extract usage from chunk and update token counters."""
        usage = chunk.get("usage")
        if not usage:
            return
        if usage.get("prompt_tokens") is not None:
            self.input_tokens = usage["prompt_tokens"]
        raw_completion = usage.get("completion_tokens")
        if raw_completion is not None:
            self.output_tokens = raw_completion
        prompt_details = usage.get("prompt_tokens_details")
        if prompt_details:
            raw_cached = prompt_details.get("cached_tokens")
            if raw_cached is not None:
                self.cached_tokens = raw_cached
            raw_creation = prompt_details.get("cache_creation_tokens")
            if raw_creation is not None:
                self.cache_creation_tokens = raw_creation

    # ------------------------------------------------------------------
    # Block lifecycle helpers
    # ------------------------------------------------------------------

    def _emit_message_start(self, current_time: float) -> list[str]:
        """Emit the message_start event (once per stream). Returns events to yield."""
        event_str = _make_message_start_event(
            self.request_id, self.original_model,
            self.input_tokens, self.cached_tokens, self.cache_creation_tokens,
        )
        self.message_started = True
        self.last_event_time = current_time
        events: list[str] = []
        if event := self._batch_add(event_str):
            events.append(event)
        return events

    def _close_thinking_block(self) -> list[str]:
        """Close the thinking block if open. Returns events to yield."""
        if not self.thinking_block_started:
            return []
        event_str = _make_content_block_stop_event(self.current_block_index)
        self.current_block_index += 1
        self.thinking_block_started = False
        events: list[str] = []
        if event := self._batch_add(event_str):
            events.append(event)
        return events

    def _close_content_block(self) -> list[str]:
        """Close the text content block if open. Returns events to yield."""
        if not self.content_block_started:
            return []
        event_str = _make_content_block_stop_event(self.current_block_index)
        self.current_block_index += 1
        self.content_block_started = False
        events: list[str] = []
        if event := self._batch_add(event_str):
            events.append(event)
        return events

    def _emit_thinking_segment(self, thinking: str, current_time: float) -> list[str]:
        """Emit one thinking segment, switching block types if needed."""
        if not thinking:
            return []

        events: list[str] = []
        if self.content_block_started:
            events.extend(self._close_content_block())

        if not self.thinking_block_started:
            event_str = _make_content_block_start_event(self.current_block_index, "thinking")
            if event := self._batch_add(event_str):
                events.append(event)
            self.thinking_block_started = True

        event_str = _make_thinking_delta_event(self.current_block_index, thinking)
        if event := self._batch_add(event_str):
            events.append(event)
        self._thinking_parts.append(thinking)
        self.last_event_time = current_time
        return events

    def _emit_text_segment(self, text: str, current_time: float) -> list[str]:
        """Emit one text segment, switching block types if needed."""
        if not text:
            return []

        events: list[str] = []
        if self.thinking_block_started:
            events.extend(self._close_thinking_block())

        if not self.content_block_started:
            text_cc = self._cache_control_map.get(self.current_block_index)
            event_str = _make_content_block_start_event(self.current_block_index, "text", cache_control=text_cc)
            if event := self._batch_add(event_str):
                events.append(event)
            self.content_block_started = True

        event_str = _make_text_delta_event(self.current_block_index, text)
        if event := self._batch_add(event_str):
            events.append(event)
        self._text_parts.append(text)
        self.last_event_time = current_time
        return events

    def _emit_content_segments(self, segments: list[tuple[str, str]], current_time: float) -> list[str]:
        """Emit parser-produced reasoning/text segments in order."""
        events: list[str] = []
        for field, text in segments:
            if field == "reasoning_content":
                events.extend(self._emit_thinking_segment(text, current_time))
            else:
                events.extend(self._emit_text_segment(text, current_time))
        return events

    def _emit_text_tool_segments(self, segments: list[tuple[str, Any]], current_time: float) -> list[str]:
        """Emit content and recovered text-form tool calls in stream order."""
        events: list[str] = []
        for field, value in segments:
            if field == "tool_call":
                events.extend(self._emit_content_segments(self._think_tag_parser.flush(), current_time))
                events.extend(self._emit_structured_tool_calls([value], current_time))
            else:
                events.extend(self._emit_content_segments(
                    self._think_tag_parser.feed(value),
                    current_time,
                ))
        return events

    # ------------------------------------------------------------------
    # Delta handlers
    # ------------------------------------------------------------------

    def _handle_thinking(self, delta: dict, current_time: float) -> list[str]:
        """Handle reasoning/thinking content delta. Returns events to yield."""
        reasoning_content = delta.get("reasoning_content")
        if not reasoning_content:
            return []
        events = self._emit_text_tool_segments(self._text_tool_parser.flush(), current_time)
        events.extend(self._emit_thinking_segment(reasoning_content, current_time))
        return events

    def _handle_text(self, delta: dict, current_time: float) -> list[str]:
        """Handle text content delta. Returns events to yield."""
        content = delta.get("content")
        if not content:
            return []
        return self._emit_text_tool_segments(self._text_tool_parser.feed(content), current_time)

    def _handle_tool_calls(self, delta: dict, current_time: float) -> list[str]:
        """Handle tool call deltas. Returns events to yield."""
        tool_calls = delta.get("tool_calls") or []
        if not tool_calls:
            return []

        events: list[str] = []
        events.extend(self._emit_text_tool_segments(self._text_tool_parser.flush(), current_time))
        events.extend(self._emit_content_segments(self._think_tag_parser.flush(), current_time))
        events.extend(self._emit_structured_tool_calls(tool_calls, current_time))
        return events

    def _emit_structured_tool_calls(self, tool_calls: list[dict], current_time: float) -> list[str]:
        """Emit OpenAI-format tool call deltas as Anthropic tool_use blocks."""
        events: list[str] = []
        for tc in tool_calls:
            tc_index = tc.get("index", 0)

            if tc_index not in self.tool_calls_by_index:
                # Close previous blocks
                events.extend(self._close_thinking_block())
                events.extend(self._close_content_block())

                # Start new tool use block
                tc_id = tc.get("id")
                if tc_id is None:
                    tc_id = f"toolu_{uuid.uuid4().hex[:12]}"
                self.tool_calls_by_index[tc_index] = {
                    "id": tc_id,
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": "",
                }
                self.tool_block_indices[tc_index] = self.current_block_index

                event_str = _make_content_block_start_event(
                    self.current_block_index,
                    "tool_use",
                    id=self.tool_calls_by_index[tc_index]["id"],
                    name=self.tool_calls_by_index[tc_index]["name"],
                    input={},
                )
                if event := self._batch_add(event_str):
                    events.append(event)
                self.current_block_index += 1

            # Accumulate arguments
            func = tc.get("function", {})
            if func.get("name"):
                self.tool_calls_by_index[tc_index]["name"] = func["name"]
            args_entry = self.tool_calls_by_index[tc_index]
            if func.get("arguments"):
                if "chunks" not in args_entry:
                    args_entry["chunks"] = []
                args_entry["chunks"].append(func["arguments"])
                event_str = _make_input_json_delta_event(
                    self.tool_block_indices[tc_index], func["arguments"]
                )
                if event := self._batch_add(event_str):
                    events.append(event)
                self.last_event_time = current_time

        return events

    # ------------------------------------------------------------------
    # Stream finalization
    # ------------------------------------------------------------------

    async def _finalize_stream(self) -> AsyncGenerator[str, None]:
        """Close all open content blocks and send final Anthropic stream events."""
        for event in self._emit_text_tool_segments(self._text_tool_parser.flush(), monotonic()):
            yield event
        for event in self._emit_content_segments(self._think_tag_parser.flush(), monotonic()):
            yield event

        batch_add = self._batch_add

        if self.thinking_block_started:
            event_str = _make_content_block_stop_event(self.current_block_index)
            if event := batch_add(event_str):
                yield event
            self.current_block_index += 1

        if self.content_block_started:
            event_str = _make_content_block_stop_event(self.current_block_index)
            if event := batch_add(event_str):
                yield event
            self.current_block_index += 1

        for tc_index in sorted(self.tool_block_indices.keys()):
            block_idx = self.tool_block_indices[tc_index]
            event_str = _make_content_block_stop_event(block_idx)
            if event := batch_add(event_str):
                yield event

        stop_reason = "tool_use" if self.tool_calls_by_index else "end_turn"

        event_str = _make_message_delta_event(stop_reason, self.output_tokens, self.cached_tokens, self.cache_creation_tokens)
        if event := batch_add(event_str):
            yield event
        event_str = _make_message_stop_event()
        if event := batch_add(event_str):
            yield event

        if remaining := self.batcher.flush():
            yield remaining

        if self.transaction_logger:
            await _log_anthropic_response(
                self.transaction_logger, self.request_id, self.original_model,
                "".join(self._thinking_parts), "".join(self._text_parts),
                self.tool_calls_by_index, self.input_tokens, self.output_tokens,
                self.cached_tokens, self.cache_creation_tokens, stop_reason,
            )

    # ------------------------------------------------------------------
    # Connection error handling
    # ------------------------------------------------------------------

    def _emit_connection_error(self, error: Exception) -> list[str]:
        """Emit error events for httpx/ConnectionError. Returns events to yield."""
        logger.error("Error in Anthropic streaming wrapper: %s", error)

        error_message = f"Error: {str(error)}"
        events: list[str] = []

        if not self.message_started:
            event_str = _make_message_start_event(
                self.request_id, self.original_model,
                self.input_tokens, self.cached_tokens, self.cache_creation_tokens,
            )
            if event := self._batch_add(event_str):
                events.append(event)
        event_str = _make_content_block_start_event(self.current_block_index, "text")
        if event := self._batch_add(event_str):
            events.append(event)
        event_str = _make_text_delta_event(self.current_block_index, error_message)
        if event := self._batch_add(event_str):
            events.append(event)
        event_str = _make_content_block_stop_event(self.current_block_index)
        if event := self._batch_add(event_str):
            events.append(event)
        event_str = _make_message_delta_event("end_turn", 0, self.cached_tokens, self.cache_creation_tokens)
        if event := self._batch_add(event_str):
            events.append(event)
        event_str = _make_message_stop_event()
        if event := self._batch_add(event_str):
            events.append(event)

        # Flush any remaining events
        if remaining := self.batcher.flush():
            events.append(remaining)

        # Send formal error event
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(error)},
        }
        events.append(f"event: error\ndata: {json_dumps(error_event)}\n\n")

        return events

    # ------------------------------------------------------------------
    # Main translation loop
    # ------------------------------------------------------------------

    async def translate(self, openai_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[str, None]:
        """Main loop: consume OpenAI SSE chunks, yield Anthropic SSE events."""
        try:
            async for raw_chunk in openai_stream:
                self._chunk_count += 1
                if (self.is_disconnected is not None
                        and self._chunk_count % self._DISCONNECT_CHECK_INTERVAL == 0
                        and await self.is_disconnected()):
                    break

                # STREAM_DONE sentinel: stream is complete
                if raw_chunk is STREAM_DONE:
                    async for event in self._finalize_stream():
                        yield event
                    break

                chunk = self._parse_raw_chunk(raw_chunk)
                if chunk is _PARSED_DONE:
                    async for event in self._finalize_stream():
                        yield event
                    break
                if chunk is None:
                    continue

                # Send heartbeat if no events for a while
                current_time = monotonic()
                if current_time - self.last_event_time > self._HEARTBEAT_INTERVAL:
                    yield ": heartbeat\n\n"
                    self.last_event_time = current_time

                # Internal retry pipeline reports terminal failures as OpenAI-style
                # error payloads. Convert them to Anthropic SSE errors.
                if "error" in chunk:
                    if remaining := self.batcher.flush():
                        yield remaining
                    yield _make_error_event(chunk.get("error"))
                    return

                self._extract_usage(chunk)

                # Send message_start on first chunk
                if not self.message_started:
                    for ev in self._emit_message_start(current_time):
                        yield ev

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                for ev in self._handle_thinking(delta, current_time):
                    yield ev
                for ev in self._handle_text(delta, current_time):
                    yield ev
                for ev in self._handle_tool_calls(delta, current_time):
                    yield ev

        except GeneratorExit:
            raise
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ConnectionError) as e:
            for ev in self._emit_connection_error(e):
                yield ev


async def anthropic_streaming_wrapper_fast(
    openai_stream: AsyncGenerator[Any, None],
    original_model: str,
    request_id: Optional[str] = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    transaction_logger: Optional["TransactionLogger"] = None,
    precomputed_input_tokens: Optional[int] = None,
    cache_control_map: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert OpenAI streaming format to Anthropic streaming format (optimized version).

    This is a framework-agnostic wrapper that can be used with any async web framework.
    Instead of taking a FastAPI Request object, it accepts an optional callback function
    to check for client disconnection.

    Anthropic SSE events:
    - message_start: Initial message metadata
    - content_block_start: Start of a content block
    - content_block_delta: Content chunk
    - content_block_stop: End of a content block
    - message_delta: Final message metadata (stop_reason, usage)
    - message_stop: End of message

    Args:
        openai_stream: AsyncGenerator yielding OpenAI chunks (dict) or SSE strings
        original_model: The model name to include in responses
        request_id: Optional request ID (auto-generated if not provided)
        is_disconnected: Optional async callback that returns True if client disconnected
        transaction_logger: Optional TransactionLogger for logging the final Anthropic response
        precomputed_input_tokens: Optional pre-computed input token count. Used as fallback
            when provider doesn't return usage in stream (e.g., Kilocode without stream_options).
            This is critical for Claude Code's context management to work correctly.
        cache_control_map: Optional mapping of content block index to cache_control dict.
            Used to restore cache_control metadata in streaming content_block_start events,
            mirroring the cache_control sent in the original Anthropic request.

    Yields:
        SSE format strings in Anthropic's streaming format
    """
    if request_id is None:
        request_id = f"msg_{uuid.uuid4().hex[:24]}"

    translator = _StreamTranslator(
        request_id=request_id,
        original_model=original_model,
        is_disconnected=is_disconnected,
        transaction_logger=transaction_logger,
        precomputed_input_tokens=precomputed_input_tokens,
        cache_control_map=cache_control_map,
    )
    async for event in translator.translate(openai_stream):
        yield event


async def _log_anthropic_response(
    transaction_logger: "TransactionLogger",
    request_id: str,
    model: str,
    accumulated_thinking: str,
    accumulated_text: str,
    tool_calls_by_index: dict,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cache_creation_tokens: int,
    stop_reason: str,
) -> None:
    """Log the final Anthropic response."""
    # Build content blocks
    content_blocks = []

    if accumulated_thinking:
        content_blocks.append({
            "type": "thinking",
            "thinking": accumulated_thinking,
        })

    if accumulated_text:
        content_blocks.append({
            "type": "text",
            "text": accumulated_text,
        })

    # Add tool use blocks
    for tc_index in sorted(tool_calls_by_index):
        tc = tool_calls_by_index[tc_index]
        try:
            chunks = tc.get("chunks")
            args_str = "".join(chunks) if chunks is not None else tc.get("arguments", "{}")
            if not args_str:
                args_str = "{}"
            input_data = json_loads(args_str)
        except (RuntimeError, ValueError, TypeError, Exception) as exc:
            logging.getLogger(__name__).debug("Suppressed: %s", exc)
            input_data = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc.get("name", ""),
            "input": input_data,
        })

    # Build usage
    log_usage = {
        "input_tokens": input_tokens - cached_tokens - cache_creation_tokens,
        "output_tokens": output_tokens,
    }
    if cached_tokens > 0:
        log_usage["cache_read_input_tokens"] = cached_tokens
    if cache_creation_tokens > 0:
        log_usage["cache_creation_input_tokens"] = cache_creation_tokens

    anthropic_response = {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": log_usage,
    }

    await transaction_logger.log_response(
        anthropic_response,
        filename="anthropic_response.json",
    )


# Export the fast version as the default wrapper
anthropic_streaming_wrapper = anthropic_streaming_wrapper_fast
