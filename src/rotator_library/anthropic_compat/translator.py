# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Format translation functions between Anthropic and OpenAI API formats.

This module provides functions to convert requests and responses between
Anthropic's Messages API format and OpenAI's Chat Completions API format.
This enables any OpenAI-compatible provider to work with Anthropic clients.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from ..error_handler import validate_response_quality
from ..utils.json_utils import json_dumps_str as _json_dumps, json_loads as _json_loads, JSONDecodeError as _json_decode_error
from .translation_audit import TranslationAuditLog

from .models import AnthropicMessagesRequest
from .streaming_fast import (
    MiniMaxTextToolCallParser,
    ThinkTagParser,
    _should_parse_text_tool_calls,
    _should_parse_think_tags,
)

_tlog = logging.getLogger("rotator_library.anthropic_compat")

MIN_THINKING_SIGNATURE_LENGTH = 100

# =============================================================================
# THINKING BUDGET TO REASONING EFFORT MAPPING
# =============================================================================

# Budget thresholds for reasoning effort levels (based on token counts)
# These map Anthropic's budget_tokens to OpenAI-style reasoning_effort levels
THINKING_BUDGET_THRESHOLDS = {
    "minimal": 4096,
    "low": 8192,
    "low_medium": 12288,
    "medium": 16384,
    "medium_high": 24576,
    "high": 32768,
}

# Providers that support granular reasoning effort levels (low_medium, medium_high, etc.)
# Other providers will receive simplified levels (low, medium, high)
GRANULAR_REASONING_PROVIDERS = {"antigravity"}


def _budget_to_reasoning_effort(budget_tokens: int, model: str) -> str:
    """
    Map Anthropic thinking budget_tokens to a reasoning_effort level.

    Args:
        budget_tokens: The thinking budget in tokens from the Anthropic request
        model: The model name (used to determine if provider supports granular levels)

    Returns:
        A reasoning_effort level string (e.g., "low", "medium", "high")
    """
    # Determine granular level based on budget
    if budget_tokens <= THINKING_BUDGET_THRESHOLDS["minimal"]:
        granular_level = "minimal"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["low"]:
        granular_level = "low"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["low_medium"]:
        granular_level = "low_medium"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["medium"]:
        granular_level = "medium"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["medium_high"]:
        granular_level = "medium_high"
    else:
        granular_level = "high"

    # Check if provider supports granular levels
    provider: str = model.split("/")[0].lower() if "/" in model else ""
    if provider in GRANULAR_REASONING_PROVIDERS:
        return granular_level

    # Simplify to basic levels for non-granular providers
    simplify_map: Dict[str, str] = {
        "minimal": "low",
        "low": "low",
        "low_medium": "medium",
        "medium": "medium",
        "medium_high": "high",
        "high": "high",
    }
    return simplify_map.get(granular_level, "medium")


def _reorder_assistant_content(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorder assistant message content blocks to ensure correct order:
    1. Thinking blocks come first (required when thinking is enabled)
    2. Text blocks come in the middle (filtering out empty ones)
    3. Tool_use blocks come at the end (required before tool_result)

    This matches Anthropic's expected ordering and prevents API errors.
    """
    if not isinstance(content, list) or len(content) <= 1:
        return content

    thinking_blocks: List[Dict[str, Any]] = []
    text_blocks: List[Dict[str, Any]] = []
    tool_use_blocks: List[Dict[str, Any]] = []
    other_blocks: List[Dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            other_blocks.append(block)
            continue

        block_type: str = block.get("type", "")

        if block_type in ("thinking", "redacted_thinking"):
            # Sanitize thinking blocks - remove cache_control and other extra fields
            sanitized: Dict[str, Any] = {
                "type": block_type,
                "thinking": block.get("thinking", ""),
            }
            if block.get("signature"):
                sanitized["signature"] = block["signature"]
            thinking_blocks.append(sanitized)

        elif block_type == "tool_use":
            tool_use_blocks.append(block)

        elif block_type == "text":
            # Only keep text blocks with meaningful content
            text: Optional[str] = block.get("text")
            if text and text.strip():
                text_blocks.append(block)

        else:
            # Other block types (images, documents, etc.) go in the text position
            other_blocks.append(block)

    # Reorder: thinking → other → text → tool_use
    return thinking_blocks + other_blocks + text_blocks + tool_use_blocks


def anthropic_to_openai_messages(
    anthropic_messages: List[Dict[str, Any]], system: Optional[Union[str, List[Dict[str, Any]]]] = None
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic message format to OpenAI format.

    Key differences:
    - Anthropic: system is a separate field, content can be string or list of blocks
    - OpenAI: system is a message with role="system", content is usually string

    Args:
        anthropic_messages: List of messages in Anthropic format
        system: Optional system message (string or list of text blocks)

    Returns:
        List of messages in OpenAI format
    """
    openai_messages: List[Dict[str, Any]] = []
    cache_control_metadata: Dict[int, Dict[str, Any]] = {}  # msg_index -> {block_idx -> cache_control}

    if all(isinstance(msg.get("content", ""), str) for msg in anthropic_messages):
        if system:
            if isinstance(system, str):
                openai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                system_text: str = " ".join(
                    block.get("text", "")
                    for block in system
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if system_text:
                    openai_messages.append({"role": "system", "content": system_text})
        for msg in anthropic_messages:
            openai_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        return openai_messages

    # Handle system message
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # System can be list of text blocks in Anthropic format
            sys_text: str = " ".join(
                block.get("text", "")
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if sys_text:
                openai_messages.append({"role": "system", "content": sys_text})

    for msg in anthropic_messages:
        role: str = msg.get("role", "user")
        content: Union[str, List[Dict[str, Any]]] = msg.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Reorder assistant content blocks to ensure correct order:
            # thinking → text → tool_use
            if role == "assistant":
                content = _reorder_assistant_content(content)

            # Handle content blocks
            openai_content: List[Dict[str, Any]] = []
            tool_calls: List[Dict[str, Any]] = []
            reasoning_parts: list[str] = []
            thinking_signature: str = ""
            # Cache control metadata: maps content block index to cache_control dict
            _cache_controls: Dict[int, Any] = {}
            _openai_content_idx: int = -1  # tracks index into openai_content

            for block in content:
                if isinstance(block, dict):
                    block_type: str = block.get("type", "text")
                    _block_cc: Optional[dict] = block.get("cache_control")

                    if block_type == "text":
                        _openai_content_idx = len(openai_content)
                        openai_content.append(
                            {"type": "text", "text": block.get("text", "")}
                        )
                        if _block_cc:
                            _cache_controls[_openai_content_idx] = _block_cc
                    elif block_type == "image":
                        # Convert Anthropic image format to OpenAI
                        source: Dict[str, Any] = block.get("source", {})
                        _openai_content_idx = len(openai_content)
                        if source.get("type") == "base64":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                    },
                                }
                            )
                        elif source.get("type") == "url":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                }
                            )
                        if _block_cc:
                            _cache_controls[_openai_content_idx] = _block_cc
                    elif block_type == "document":
                        # Convert Anthropic document format (e.g. PDF) to OpenAI
                        # Documents are treated similarly to images with appropriate mime type
                        doc_source: Dict[str, Any] = block.get("source", {})
                        _openai_content_idx = len(openai_content)
                        if doc_source.get("type") == "base64":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{doc_source.get('media_type', 'application/pdf')};base64,{doc_source.get('data', '')}"
                                    },
                                }
                            )
                        elif doc_source.get("type") == "url":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": doc_source.get("url", "")},
                                }
                            )
                        if _block_cc:
                            _cache_controls[_openai_content_idx] = _block_cc
                    elif block_type == "thinking":
                        signature: str = block.get("signature", "")
                        thinking_text: str = block.get("thinking", "")
                        if thinking_text:
                            reasoning_parts.append(thinking_text)
                        if (
                            signature
                            and len(signature) >= MIN_THINKING_SIGNATURE_LENGTH
                        ):
                            thinking_signature = signature
                    elif block_type == "redacted_thinking":
                        redacted_sig: str = block.get("signature", "")
                        if (
                            redacted_sig
                            and len(redacted_sig) >= MIN_THINKING_SIGNATURE_LENGTH
                        ):
                            thinking_signature = redacted_sig
                    elif block_type == "tool_use":
                        # Anthropic tool_use -> OpenAI tool_calls
                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": _json_dumps(block.get("input", {})),
                                },
                            }
                        )
                    elif block_type == "tool_result":
                        # Tool results become separate messages in OpenAI format
                        # Content can be string, or list of text/image blocks
                        tool_content: Union[str, List[Dict[str, Any]]] = block.get("content", "")
                        if isinstance(tool_content, str):
                            # Simple string content
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": tool_content,
                                }
                            )
                        elif isinstance(tool_content, list):
                            # List of content blocks - may include text and images
                            tool_content_parts: List[Dict[str, Any]] = []
                            for b in tool_content:
                                if not isinstance(b, dict):
                                    continue
                                b_type: str = b.get("type", "")
                                if b_type == "text":
                                    tool_content_parts.append(
                                        {"type": "text", "text": b.get("text", "")}
                                    )
                                elif b_type == "image":
                                    # Convert Anthropic image format to OpenAI format
                                    tool_source: Dict[str, Any] = b.get("source", {})
                                    if tool_source.get("type") == "base64":
                                        tool_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{tool_source.get('media_type', 'image/png')};base64,{tool_source.get('data', '')}"
                                                },
                                            }
                                        )
                                    elif tool_source.get("type") == "url":
                                        tool_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": tool_source.get("url", "")
                                                },
                                            }
                                        )

                            # If we only have text parts, join them as a string for compatibility
                            # Otherwise use the array format for multimodal content
                            tool_texts: List[str] = []
                            all_text = True
                            for p in tool_content_parts:
                                if p.get("type") != "text":
                                    all_text = False
                                    break
                                tool_texts.append(p.get("text", ""))
                            if all_text and tool_texts:
                                combined_text: str = " ".join(tool_texts)
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": combined_text,
                                    }
                                )
                            elif tool_content_parts:
                                # Multimodal content (includes images)
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": tool_content_parts,
                                    }
                                )
                            else:
                                # Empty content
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": "",
                                    }
                                )
                        else:
                            # Fallback for unexpected content type
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(tool_content)
                                    if tool_content
                                    else "",
                                }
                            )
                        continue  # Don't add to current message

            # Build the message
            reasoning_content: str = "".join(reasoning_parts) if reasoning_parts else ""
            if tool_calls:
                # Assistant message with tool calls
                msg_dict: Dict[str, Any] = {"role": role}
                if openai_content:
                    # If there's text content alongside tool calls
                    text_parts: List[str] = [
                        c.get("text", "")
                        for c in openai_content
                        if c.get("type") == "text"
                    ]
                    msg_dict["content"] = " ".join(text_parts) if text_parts else ""
                else:
                    msg_dict["content"] = ""
                if reasoning_content:
                    msg_dict["reasoning_content"] = reasoning_content
                if thinking_signature:
                    msg_dict["thinking_signature"] = thinking_signature
                msg_dict["tool_calls"] = tool_calls
                if _cache_controls:
                    msg_dict["_cache_control_metadata"] = _cache_controls
                openai_messages.append(msg_dict)
            elif openai_content:
                # Check if it's just text or mixed content
                if len(openai_content) == 1 and openai_content[0].get("type") == "text":
                    msg_dict = {
                        "role": role,
                        "content": openai_content[0].get("text", ""),
                    }
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
                    if thinking_signature:
                        msg_dict["thinking_signature"] = thinking_signature
                    if _cache_controls:
                        msg_dict["_cache_control_metadata"] = _cache_controls
                    openai_messages.append(msg_dict)
                else:
                    msg_dict = {"role": role, "content": openai_content}
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
                    if thinking_signature:
                        msg_dict["thinking_signature"] = thinking_signature
                    if _cache_controls:
                        msg_dict["_cache_control_metadata"] = _cache_controls
                    openai_messages.append(msg_dict)
            elif reasoning_content:
                msg_dict = {"role": role, "content": ""}
                msg_dict["reasoning_content"] = reasoning_content
                if thinking_signature:
                    msg_dict["thinking_signature"] = thinking_signature
                openai_messages.append(msg_dict)

    return openai_messages


def anthropic_to_openai_tools(
    anthropic_tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Anthropic tool definitions to OpenAI format.

    Args:
        anthropic_tools: List of tools in Anthropic format

    Returns:
        List of tools in OpenAI format, or None if no tools provided
    """
    if not anthropic_tools:
        return None

    openai_tools: List[Dict[str, Any]] = []
    for tool in anthropic_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return openai_tools


def anthropic_to_openai_tool_choice(
    anthropic_tool_choice: Optional[Union[str, Dict[str, Any]]],
) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Args:
        anthropic_tool_choice: Tool choice in Anthropic format (str or dict)

    Returns:
        Tool choice in OpenAI format
    """
    if not anthropic_tool_choice:
        return None

    if isinstance(anthropic_tool_choice, str):
        if anthropic_tool_choice == "any":
            return "required"
        elif anthropic_tool_choice == "none":
            return "none"
        return "auto"

    choice_type: str = anthropic_tool_choice.get("type", "auto")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": anthropic_tool_choice.get("name", "")},
        }
    elif choice_type == "none":
        return "none"

    return "auto"


def _openai_tool_call_to_anthropic_block(tc: Dict[str, Any]) -> Dict[str, Any]:
    func: Dict[str, Any] = tc.get("function", {})
    try:
        input_data: Any = _json_loads(func.get("arguments", "{}"))
    except _json_decode_error:
        input_data = {}

    tc_id = tc.get("id")
    if tc_id is None:
        tc_id = f"toolu_{uuid.uuid4().hex[:12]}"

    return {
        "type": "tool_use",
        "id": tc_id,
        "name": func.get("name", ""),
        "input": input_data,
    }


def openai_to_anthropic_response(openai_response: Dict[str, Any], original_model: str) -> Dict[str, Any]:
    """
    Convert OpenAI chat completion response to Anthropic Messages format.

    Args:
        openai_response: Response from OpenAI-compatible API
        original_model: The model name requested by the client

    Returns:
        Response in Anthropic Messages format
    """
    choice: Dict[str, Any] = openai_response.get("choices", [{}])[0]
    message: Dict[str, Any] = choice.get("message", {})
    usage: Dict[str, Any] = openai_response.get("usage", {})

    # Build content blocks
    content_blocks: List[Dict[str, Any]] = []

    # Add thinking content block if reasoning_content is present
    reasoning_content: Optional[str] = message.get("reasoning_content")
    if reasoning_content:
        thinking_signature: Optional[str] = message.get("thinking_signature")
        signature: str = (
            thinking_signature
            if thinking_signature
            and len(thinking_signature) >= MIN_THINKING_SIGNATURE_LENGTH
            else ""
        )
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning_content,
                "signature": signature,
            }
        )

    recovered_tool_call_seen = False

    # Add text content if present
    text_content: Optional[str] = message.get("content")
    if text_content:
        if _should_parse_text_tool_calls(original_model):
            tool_parser = MiniMaxTextToolCallParser(enabled=True)
            think_parser = ThinkTagParser(enabled=_should_parse_think_tags(original_model))
            for field, value in tool_parser.feed(text_content, final=True):
                if field == "tool_call":
                    for sub_field, sub_text in think_parser.flush():
                        if not sub_text:
                            continue
                        if sub_field == "reasoning_content":
                            content_blocks.append({"type": "thinking", "thinking": sub_text})
                        else:
                            content_blocks.append({"type": "text", "text": sub_text})
                    content_blocks.append(_openai_tool_call_to_anthropic_block(value))
                    recovered_tool_call_seen = True
                    continue
                for sub_field, sub_text in think_parser.feed(value):
                    if not sub_text:
                        continue
                    if sub_field == "reasoning_content":
                        content_blocks.append({"type": "thinking", "thinking": sub_text})
                    else:
                        content_blocks.append({"type": "text", "text": sub_text})
            for sub_field, sub_text in think_parser.flush():
                if not sub_text:
                    continue
                if sub_field == "reasoning_content":
                    content_blocks.append({"type": "thinking", "thinking": sub_text})
                else:
                    content_blocks.append({"type": "text", "text": sub_text})
        elif _should_parse_think_tags(original_model):
            parser = ThinkTagParser(enabled=True)
            for field, text in parser.feed(text_content, final=True):
                if not text:
                    continue
                if field == "reasoning_content":
                    content_blocks.append({"type": "thinking", "thinking": text})
                else:
                    content_blocks.append({"type": "text", "text": text})
        else:
            content_blocks.append({"type": "text", "text": text_content})

    # Add structured tool use blocks if present.
    tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []
    for tc in tool_calls:
        content_blocks.append(_openai_tool_call_to_anthropic_block(tc))

    # Map finish_reason to stop_reason
    finish_reason: str = choice.get("finish_reason", "end_turn")
    stop_reason_map: Dict[str, str] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }
    stop_reason: str = (
        "tool_use"
        if recovered_tool_call_seen or tool_calls
        else stop_reason_map.get(finish_reason, "end_turn")
    )

    # Build usage
    # Note: Google's promptTokenCount INCLUDES cached tokens, but Anthropic's
    # input_tokens EXCLUDES cached tokens. We need to subtract cached tokens.
    prompt_tokens: int = usage.get("prompt_tokens") or 0
    cached_tokens: int = 0

    # Extract cached tokens if present
    if usage.get("prompt_tokens_details"):
        details: Dict[str, int] = usage["prompt_tokens_details"]
        cached_tokens = details.get("cached_tokens") or 0

    anthropic_usage: Dict[str, int] = {
        "input_tokens": prompt_tokens - cached_tokens,
        "output_tokens": usage.get("completion_tokens", 0),
    }

    # Add cache tokens if present
    if cached_tokens > 0:
        anthropic_usage["cache_read_input_tokens"] = cached_tokens
        anthropic_usage["cache_creation_input_tokens"] = 0

    # Validate response quality before translating to Anthropic format
    # GarbageResponseError propagates to the retry loop in _retry.py for credential rotation
    provider: str = original_model.split("/")[0] if "/" in original_model else ""
    validate_response_quality(openai_response, provider=provider, model=original_model)

    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }


def translate_anthropic_request(request: AnthropicMessagesRequest) -> Dict[str, Any]:
    """
    Translate a complete Anthropic Messages API request to OpenAI format.

    This is a high-level function that handles all aspects of request translation,
    including messages, tools, tool_choice, and thinking configuration.

    Args:
        request: An AnthropicMessagesRequest object

    Returns:
        Dictionary containing the OpenAI-compatible request parameters
    """
    messages = [m.model_dump(exclude_none=True) for m in request.messages]

    openai_messages: List[Dict[str, Any]] = anthropic_to_openai_messages(messages, request.system)

    tools = [t.model_dump(exclude_none=True) for t in request.tools] if request.tools else None
    tool_choice = request.tool_choice

    openai_tools: Optional[List[Dict[str, Any]]] = anthropic_to_openai_tools(tools)
    openai_tool_choice: Optional[Union[str, Dict[str, Any]]] = anthropic_to_openai_tool_choice(tool_choice)

    # Build OpenAI-compatible request
    openai_request: Dict[str, Any] = {
        "model": request.model,
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream or False,
    }

    # Audit tracking fields
    _preserved: List[str] = []
    _dropped: List[str] = []
    _transformed: Dict[str, Any] = {}

    if request.temperature is not None:
        openai_request["temperature"] = request.temperature
        _preserved.append("temperature")
    if request.top_p is not None:
        openai_request["top_p"] = request.top_p
        _preserved.append("top_p")
    if request.top_k is not None:
        # top_k is Anthropic-specific; pass through but log it.
        # Most OpenAI-compatible providers silently ignore it.
        openai_request["top_k"] = request.top_k
        _tlog.debug(
            "top_k=%d passed through for model %s (provider may ignore it)",
            request.top_k, request.model,
        )
        _transformed["top_k"] = {"before": request.top_k, "after": request.top_k}
    if request.stop_sequences:
        openai_request["stop"] = request.stop_sequences
        _transformed["stop_sequences"] = {"before": "stop_sequences", "after": "stop"}
    if openai_tools:
        openai_request["tools"] = openai_tools
        _preserved.append("tools")
    if openai_tool_choice:
        openai_request["tool_choice"] = openai_tool_choice
        _transformed["tool_choice"] = {"before": tool_choice, "after": openai_tool_choice}

    # Note: request.metadata is intentionally not mapped.
    # OpenAI's API doesn't have an equivalent field for client-side metadata.
    # The metadata is typically used by Anthropic clients for tracking purposes
    # and doesn't affect the model's behavior.

    # Handle Anthropic thinking config -> reasoning_effort translation
    # Only set reasoning_effort if thinking is explicitly configured
    thinking: Any = request.thinking
    if thinking:
        _thinking_type: Optional[str] = None
        _budget_tokens: Optional[int] = None
        if isinstance(thinking, dict):
            _thinking_type = thinking.get("type")
            _budget_tokens = thinking.get("budget_tokens")
        else:
            _thinking_type = getattr(thinking, "type", None)
            _budget_tokens = getattr(thinking, "budget_tokens", None)
        if _thinking_type == "enabled":
            if _budget_tokens is not None:
                effort = _budget_to_reasoning_effort(_budget_tokens, request.model)
                openai_request["reasoning_effort"] = effort
                _transformed["thinking"] = {
                    "before": {"type": "enabled", "budget_tokens": _budget_tokens},
                    "after": effort,
                }
        elif _thinking_type == "disabled":
            openai_request["reasoning_effort"] = "disable"
            _transformed["thinking"] = {"before": {"type": "disabled"}, "after": "disable"}

    # Check if any messages carry cache_control metadata
    _has_cache_control = any(
        "_cache_control_metadata" in m for m in openai_messages
    )
    if _has_cache_control:
        _preserved.append("cache_control")

    # Embed audit metadata in request for downstream consumers
    openai_request["_translation_audit"] = {
        "fields_preserved": _preserved,
        "fields_dropped": _dropped,
        "fields_transformed": _transformed,
    }

    return openai_request


def openai_to_anthropic_tool_choice(
    openai_tool_choice: Optional[Union[str, Dict[str, Any]]],
) -> Optional[Union[str, Dict[str, Any]]]:
    """Reverse of :func:`anthropic_to_openai_tool_choice`.

    Maps OpenAI tool_choice back to Anthropic format.

    Args:
        openai_tool_choice: Tool choice in OpenAI format

    Returns:
        Tool choice in Anthropic format, or None if not set
    """
    if not openai_tool_choice:
        return None

    if isinstance(openai_tool_choice, str):
        if openai_tool_choice == "required":
            return "any"
        elif openai_tool_choice == "none":
            return "none"
        return "auto"

    if isinstance(openai_tool_choice, dict):
        if openai_tool_choice.get("type") == "function":
            func = openai_tool_choice.get("function", {})
            return {"type": "tool", "name": func.get("name", "")}

    return "auto"


def openai_to_anthropic_messages(
    openai_messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[Union[str, List[Dict[str, Any]]]]]:
    """
    Convert OpenAI message format back to Anthropic format with cache_control restoration.

    This is the reverse of :func:`anthropic_to_openai_messages`. It restores
    cache_control metadata that was preserved during forward translation.

    Args:
        openai_messages: List of messages in OpenAI format

    Returns:
        Tuple of (anthropic_messages, system) where:
        - anthropic_messages: List of messages in Anthropic format
        - system: System message content (string or list of blocks), or None
    """
    anthropic_messages: List[Dict[str, Any]] = []
    system: Optional[Union[str, List[Dict[str, Any]]]] = None

    for msg in openai_messages:
        role: str = msg.get("role", "user")
        content: Union[str, List[Dict[str, Any]]] = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system = content
            elif isinstance(content, list):
                system = [{"type": "text", "text": c.get("text", "")}
                         for c in content if c.get("type") == "text"]
            continue

        cache_control_metadata = msg.get("_cache_control_metadata", {})

        if isinstance(content, str):
            if role == "tool":
                anthropic_messages.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                })
            elif cache_control_metadata:
                # Restore list format when cache_control metadata is present
                # This happens when forward translator flattened a single text block
                anthropic_messages.append({
                    "role": role,
                    "content": [{
                        "type": "text",
                        "text": content,
                        "cache_control": cache_control_metadata.get(0),
                    }]
                })
            else:
                anthropic_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            anthropic_content: List[Dict[str, Any]] = []
            openai_idx = 0

            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type: str = block.get("type", "text")

                if block_type == "text":
                    text_block: Dict[str, Any] = {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                    if openai_idx in cache_control_metadata:
                        text_block["cache_control"] = cache_control_metadata[openai_idx]
                    anthropic_content.append(text_block)
                    openai_idx += 1
                elif block_type == "image_url":
                    image_url: str = block.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        parts = image_url.split(",", 1)
                        media_type = parts[0].split(";")[0].replace("data:", "")
                        image_block: Dict[str, Any] = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": parts[1] if len(parts) > 1 else "",
                            }
                        }
                        if openai_idx in cache_control_metadata:
                            image_block["cache_control"] = cache_control_metadata[openai_idx]
                        anthropic_content.append(image_block)
                    else:
                        image_block = {
                            "type": "image",
                            "source": {"type": "url", "url": image_url}
                        }
                        if openai_idx in cache_control_metadata:
                            image_block["cache_control"] = cache_control_metadata[openai_idx]
                        anthropic_content.append(image_block)
                    openai_idx += 1

            if role == "tool":
                anthropic_messages.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": anthropic_content,
                })
            else:
                anthropic_messages.append({"role": role, "content": anthropic_content})

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                try:
                    input_data = _json_loads(func.get("arguments", "{}"))
                except _json_decode_error:
                    input_data = {}
                anthropic_messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": input_data,
                    }]
                })

    return anthropic_messages, system
