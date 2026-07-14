# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Anthropic API compatibility module for rotator_library.

This module provides format translation between Anthropic's Messages API
and OpenAI's Chat Completions API, enabling any OpenAI-compatible provider
to work with Anthropic clients like Claude Code.

Usage:
    from rotator_library.anthropic_compat import (
        AnthropicMessagesRequest,
        translate_anthropic_request,
        openai_to_anthropic_response,
        anthropic_streaming_wrapper,
    )
"""

from .models import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)

from .translator import (
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    anthropic_to_openai_tool_choice,
    openai_to_anthropic_messages,
    openai_to_anthropic_response,
    openai_to_anthropic_tool_choice,
    translate_anthropic_request,
)

from .streaming_fast import anthropic_streaming_wrapper, build_tool_schema_map
from .translation_audit import TranslationAuditLog

__all__ = [
    # Models
    "AnthropicMessagesRequest",
    "AnthropicCountTokensRequest",
    # Translator functions
    "anthropic_to_openai_messages",
    "anthropic_to_openai_tools",
    "anthropic_to_openai_tool_choice",
    "openai_to_anthropic_messages",
    "openai_to_anthropic_response",
    "openai_to_anthropic_tool_choice",
    "translate_anthropic_request",
    # Streaming
    "anthropic_streaming_wrapper",
    "build_tool_schema_map",
    # Audit
    "TranslationAuditLog",
]
