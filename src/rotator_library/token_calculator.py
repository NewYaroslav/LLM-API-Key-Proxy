# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Automatic max_tokens calculation to prevent context window overflow errors.

This module calculates balanced max_tokens values based on:
1. Model's context window limit (from ModelRegistry)
2. Current input token count (messages + tools)
3. Safety buffer to avoid edge cases
"""

import functools
import logging
import os
import re
from typing import Dict, Any, Optional, Tuple

from .utils.json_utils import json_dumps_str

from litellm.litellm_core_utils.token_counter import token_counter  # type: ignore[import-untyped]

logger = logging.getLogger("rotator_library")

try:
    EXACT_TOKEN_COUNTER_MAX_CHARS = int(
        os.getenv("EXACT_TOKEN_COUNTER_MAX_CHARS", "250000")
    )
except ValueError:
    EXACT_TOKEN_COUNTER_MAX_CHARS = 250000

# Models that require `max_completion_tokens` exclusively — sending both
# `max_tokens` and `max_completion_tokens` produces an upstream 400.
_MAX_COMPLETION_TOKENS_MODEL_PREFIXES: tuple = (
    "openai/",
    "gpt-5",
    "gpt-image",
    "o1-",
    "o3-",
    "o4-",
)


def _normalize_max_tokens_keys(payload: dict, model: str) -> None:
    """Drop `max_tokens` when both keys exist for models that require
    `max_completion_tokens` exclusively. Mutates payload in place."""
    if not model:
        return
    model_lower = model.lower()
    if not model_lower.startswith(_MAX_COMPLETION_TOKENS_MODEL_PREFIXES):
        return
    if "max_tokens" in payload and "max_completion_tokens" in payload:
        dropped = payload.pop("max_tokens", None)
        logger.debug(
            "Normalized max_tokens keys for model %s: dropped max_tokens=%s, "
            "kept max_completion_tokens=%s",
            model,
            dropped,
            payload.get("max_completion_tokens"),
        )


# Default context window sizes for common models (fallback when registry unavailable)
DEFAULT_CONTEXT_WINDOWS: Dict[str, int] = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3.5-haiku": 200000,
    "claude-sonnet-4": 200000,
    "claude-opus-4": 200000,
    # Google
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash": 1048576,
    # DeepSeek
    "deepseek-chat": 64000,
    "deepseek-coder": 64000,
    "deepseek-reasoner": 64000,
    # Mistral
    "mistral-large": 128000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    # Other common
    "llama-3.1-405b": 131072,
    "llama-3.1-70b": 131072,
    "llama-3.1-8b": 131072,
    # ZhipuAI / GLM models (via Kilocode, Z-AI, etc.)
    "glm-4": 128000,
    "glm-4-plus": 128000,
    "glm-4-air": 128000,
    "glm-4-flash": 128000,
    "glm-5": 202800,
    "z-ai/glm-5": 202800,
}

# Fallback maximum output tokens for models where external catalogs are
# inaccurate or unavailable.  These are applied as a hard cap on top of the
# context-window-based calculation.
DEFAULT_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    # Moonshot / Kimi
    "kimi-k2.6": 98304,
    # ZhipuAI / GLM
    "glm-5.1": 131072,
}

# Safety buffer (tokens reserved for system overhead, response formatting, etc.)
# Increased from 100 to 1000 to account for:
# - Token counting estimation errors (~5-10%)
# - Provider-specific tokenization differences
# - System message and metadata overhead
# - Tool definitions and function call overhead
DEFAULT_SAFETY_BUFFER = 1000

# Minimum max_tokens to request (avoid degenerate cases)
MIN_MAX_TOKENS = 256

# Maximum percentage of context window for input (leave room for output)
# If input exceeds this, messages should be trimmed or request rejected
MAX_INPUT_RATIO = 1.0

# Extra buffer for providers with known tokenization differences
PROVIDER_SAFETY_BUFFERS = {
    "kilocode": 2000,  # Kilocode has additional overhead
    "openrouter": 1500,
    "gemini": 1000,  # Gemini tokenization can vary
    "anthropic": 500,
}


def extract_model_name(model: str) -> str:
    """
    Extract the base model name from a provider-prefixed model string.

    Examples:
        "openai/gpt-4o" -> "gpt-4o"
        "anthropic/claude-3-opus" -> "claude-3-opus"
        "kilocode/z-ai/glm-5:free" -> "z-ai/glm-5:free"
    """
    if "/" in model:
        parts = model.split("/", 1)
        return parts[1] if len(parts) > 1 else model
    return model


_RE_DATE_SUFFIX = re.compile(r"-[0-9]{4,}$")
_RE_PREVIEW_SUFFIX = re.compile(r"-preview$")
_RE_LATEST_SUFFIX = re.compile(r"-latest$")


@functools.lru_cache(maxsize=256)
def normalize_model_name(model: str) -> str:
    """
    Normalize model name for lookup.

    Handles common variations like:
        "gpt-4-0125-preview" -> "gpt-4-turbo"
        "claude-3-opus-20240229" -> "claude-3-opus"
    """
    model = model.lower().strip()

    # Remove version/date suffixes
    model = _RE_DATE_SUFFIX.sub("", model)
    model = _RE_PREVIEW_SUFFIX.sub("", model)
    model = _RE_LATEST_SUFFIX.sub("", model)

    return model


def get_context_window(model: str, registry=None) -> Optional[int]:
    """
    Get the context window size for a model.

    Args:
        model: Full model identifier (e.g., "openai/gpt-4o")
        registry: Optional ModelRegistry instance for lookups

    Returns:
        Context window size in tokens, or None if unknown
    """
    # Try registry first if available
    if registry is not None:
        try:
            metadata = registry.lookup(model)
            if metadata and metadata.limits.context_window:
                return metadata.limits.context_window
        except (ValueError, KeyError, TypeError, Exception) as e:
            logger.debug(f"Registry lookup failed for {model}: {e}")

    # Extract base model name
    base_model = extract_model_name(model)
    normalized = normalize_model_name(base_model)

    # Try direct match
    if base_model in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[base_model]

    if normalized in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[normalized]

    # Try partial matches
    for pattern, window in DEFAULT_CONTEXT_WINDOWS.items():
        if pattern in normalized or normalized in pattern:
            return window

    # Special handling for common prefixes
    for prefix in ["gpt-4", "gpt-3.5", "claude-3", "gemini-", "deepseek", "mistral"]:
        if normalized.startswith(prefix):
            for pattern, window in DEFAULT_CONTEXT_WINDOWS.items():
                if pattern.startswith(prefix):
                    return window

    return None


def _estimate_text_tokens(text: str) -> int:
    """Fast local token estimate used when exact tokenizers are too expensive."""
    if not text:
        return 0

    ascii_chars = 0
    non_ascii_chars = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1

    # ASCII text is usually ~4 chars/token; non-ASCII often trends closer to
    # 1 char/token. This intentionally errs on the conservative side.
    return max(1, (ascii_chars + 3) // 4 + non_ascii_chars)


def _estimate_value_tokens(value: Any) -> tuple[int, int]:
    """Return (approx_tokens, approx_chars) for nested JSON-like payloads."""
    if value is None:
        return 0, 0
    if isinstance(value, str):
        return _estimate_text_tokens(value), len(value)
    if isinstance(value, (int, float, bool)):
        text = str(value)
        return _estimate_text_tokens(text), len(text)
    if isinstance(value, dict):
        tokens = 4
        chars = 0
        for key, item in value.items():
            key_tokens, key_chars = _estimate_value_tokens(key)
            item_tokens, item_chars = _estimate_value_tokens(item)
            tokens += key_tokens + item_tokens + 2
            chars += key_chars + item_chars
        return tokens, chars
    if isinstance(value, (list, tuple)):
        tokens = 2
        chars = 0
        for item in value:
            item_tokens, item_chars = _estimate_value_tokens(item)
            tokens += item_tokens + 1
            chars += item_chars
        return tokens, chars

    text = str(value)
    return _estimate_text_tokens(text), len(text)


def estimate_input_tokens(
    messages: Optional[list] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[Any] = None,
) -> tuple[int, int]:
    """Cheap token estimate for request sizing.

    Returns:
        Tuple of (estimated_tokens, estimated_chars).
    """
    tokens = 0
    chars = 0
    for value in (messages, tools, tool_choice):
        value_tokens, value_chars = _estimate_value_tokens(value)
        tokens += value_tokens
        chars += value_chars
    return tokens, chars


def count_input_tokens(
    messages: list,
    model: str,
    tools: Optional[list] = None,
    tool_choice: Optional[Any] = None,
) -> int:
    """
    Count total input tokens including messages and tools.

    Args:
        messages: List of message dictionaries
        model: Model identifier for token counting
        tools: Optional list of tool definitions
        tool_choice: Optional tool choice parameter

    Returns:
        Total input token count
    """
    total = 0
    message_estimate, message_chars = estimate_input_tokens(messages=messages)
    tool_estimate, tool_chars = estimate_input_tokens(
        tools=tools, tool_choice=tool_choice
    )
    estimated_tokens = message_estimate + tool_estimate
    estimated_chars = message_chars + tool_chars

    if estimated_chars > EXACT_TOKEN_COUNTER_MAX_CHARS:
        logger.debug(
            "Using approximate token count for large payload: model=%s chars=%d tokens~=%d threshold=%d",
            model,
            estimated_chars,
            estimated_tokens,
            EXACT_TOKEN_COUNTER_MAX_CHARS,
        )
        return estimated_tokens

    # Count message tokens
    if messages:
        try:
            total += token_counter(model=model, messages=messages)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to count message tokens: {e}")
            total += message_estimate

    # Count tool definition tokens
    if tools:
        try:
            tools_json = json_dumps_str(tools)
            total += token_counter(model=model, text=tools_json)
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Failed to count tool tokens: {e}")
            total += tool_estimate

    return total


def get_max_output_tokens(model: str, registry=None) -> Optional[int]:
    """
    Get the maximum output tokens for a model.

    Tries the provider registry first, then falls back to DEFAULT_MAX_OUTPUT_TOKENS.

    Args:
        model: Full model identifier (e.g., "openai/gpt-5.5")
        registry: ModelRegistry instance for lookups

    Returns:
        Maximum output tokens, or None if unknown
    """
    # Try registry first if available
    if registry is not None:
        try:
            metadata = registry.lookup(model)
            if metadata and metadata.limits.max_output:
                return metadata.limits.max_output
        except (ValueError, KeyError, TypeError, Exception) as e:
            logger.debug(f"Registry lookup failed for {model}: {e}")

    # Fallback to static catalog
    base_model = extract_model_name(model)
    normalized = normalize_model_name(base_model)

    if base_model in DEFAULT_MAX_OUTPUT_TOKENS:
        return DEFAULT_MAX_OUTPUT_TOKENS[base_model]

    if normalized in DEFAULT_MAX_OUTPUT_TOKENS:
        return DEFAULT_MAX_OUTPUT_TOKENS[normalized]

    # Partial / prefix matches
    for pattern, limit in DEFAULT_MAX_OUTPUT_TOKENS.items():
        if pattern in normalized or normalized in pattern:
            return limit

    return None


def get_provider_safety_buffer(model: str) -> int:
    """
    Get provider-specific safety buffer based on model prefix.

    Args:
        model: Full model identifier (e.g., "kilocode/z-ai/glm-5:free")

    Returns:
        Safety buffer for this provider
    """
    # Extract provider from model
    if "/" in model:
        provider = model.split("/")[0].lower()
        if provider in PROVIDER_SAFETY_BUFFERS:
            return PROVIDER_SAFETY_BUFFERS[provider]
    return DEFAULT_SAFETY_BUFFER


def calculate_max_tokens(
    model: str,
    messages: Optional[list] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[Any] = None,
    requested_max_tokens: Optional[int] = None,
    registry=None,
    safety_buffer: Optional[int] = None,
) -> Tuple[Optional[int], str]:
    """
    Calculate a safe max_tokens value based on context window and input.

    Args:
        model: Full model identifier
        messages: List of message dictionaries
        tools: Optional list of tool definitions
        tool_choice: Optional tool choice parameter
        requested_max_tokens: User-requested max_tokens (if any)
        registry: Optional ModelRegistry for context window lookup
        safety_buffer: Extra buffer for safety (default: auto-detect from provider)

    Returns:
        Tuple of (calculated_max_tokens, reason) where reason explains the calculation
        Returns (None, "input_exceeds_context") if input is too large and cannot be processed
    """
    # Get context window
    context_window = get_context_window(model, registry)

    if context_window is None:
        if requested_max_tokens is not None:
            return requested_max_tokens, "unknown_context_window_using_requested"
        return None, "unknown_context_window_no_request"

    # Use provider-specific buffer if not specified
    if safety_buffer is None:
        safety_buffer = get_provider_safety_buffer(model)

    # Count input tokens
    input_tokens = 0
    if messages:
        input_tokens = count_input_tokens(messages, model, tools, tool_choice)

    # CRITICAL CHECK: Input must not exceed max allowed ratio
    max_input_allowed = int(context_window * MAX_INPUT_RATIO)
    if input_tokens > max_input_allowed:
        logger.error(
            f"Input tokens ({input_tokens}) exceed maximum allowed ({max_input_allowed}) "
            f"for context window ({context_window}). Model: {model}. "
            f"Request will fail - consider reducing conversation history."
        )
        # Return None to signal the request should be rejected
        return (
            None,
            f"input_exceeds_context_by_{input_tokens - max_input_allowed}_tokens",
        )

    # Calculate available space for output
    available_for_output = context_window - input_tokens - safety_buffer

    if available_for_output < MIN_MAX_TOKENS:
        # Input is too large - log warning but allow minimal response
        logger.warning(
            f"Input tokens ({input_tokens}) leave insufficient space for output "
            f"(available: {available_for_output}, min: {MIN_MAX_TOKENS}). Model: {model}"
        )
        return MIN_MAX_TOKENS, "input_exceeds_context_minimal_output"

    # Hard cap by model-specific max_output when known
    max_output = get_max_output_tokens(model, registry)
    if max_output is not None and available_for_output > max_output:
        capped_available = max_output
    else:
        capped_available = available_for_output

    # If user requested a specific value, honor it if valid
    if requested_max_tokens is not None:
        if requested_max_tokens <= capped_available:
            return requested_max_tokens, "using_requested_within_limit"
        else:
            # User requested too much, cap it
            return (
                capped_available,
                f"capped_from_{requested_max_tokens}_to_{capped_available}",
            )

    # No specific request - use calculated value
    if max_output is not None and capped_available == max_output:
        reason = (
            f"output_capped_to_{max_output}_from_"
            f"context_{context_window}_input_{input_tokens}"
        )
    else:
        reason = f"calculated_from_context_{context_window}_input_{input_tokens}"
    return (capped_available, reason)


def adjust_max_tokens_in_payload(
    payload: Dict[str, Any],
    model: str,
    registry=None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Adjust max_tokens in a request payload to prevent context overflow.

    This function:
    1. Calculates input token count from messages + tools
    2. Gets context window for the model
    3. Sets max_tokens to a safe value if not already set or if too large
    4. Returns flag indicating if request should be rejected

    Args:
        payload: Request payload dictionary
        model: Model identifier
        registry: Optional ModelRegistry instance

    Returns:
        Tuple of (modified payload, should_reject flag)
        If should_reject is True, the input exceeds context window and request will fail
    """
    # Check if max_tokens adjustment is needed
    # Look for both max_tokens (OpenAI) and max_completion_tokens (newer OpenAI)
    requested_max = payload.get("max_tokens") or payload.get("max_completion_tokens")

    messages = payload.get("messages", [])
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")

    # Calculate safe max_tokens
    calculated_max, reason = calculate_max_tokens(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        requested_max_tokens=requested_max,
        registry=registry,
    )

    # Check if request should be rejected due to input exceeding context
    if calculated_max is None and "input_exceeds_context" in reason:
        logger.error(
            f"Rejecting request for {model}: {reason}. "
            f"Input tokens exceed context window capacity."
        )
        return payload, True  # Signal to reject

    if calculated_max is not None:
        # Log the adjustment
        if requested_max is None:
            logger.info(
                f"Auto-setting max_tokens={calculated_max} for model {model} "
                f"(reason: {reason})"
            )
        elif calculated_max != requested_max:
            logger.info(
                f"Adjusting max_tokens from {requested_max} to {calculated_max} "
                f"for model {model} (reason: {reason})"
            )

        # OpenAI models use max_completion_tokens exclusively — setting both causes 400
        if "max_completion_tokens" in payload or model.startswith(("openai/", "gpt")):
            payload.pop("max_tokens", None)
            payload["max_completion_tokens"] = calculated_max
        else:
            payload["max_tokens"] = calculated_max

    # Unconditional safety net: if a caller supplied `max_tokens` AND another
    # code path injected `max_completion_tokens`, strip `max_tokens` for models
    # that reject the dual-key payload (upstream 400).
    _normalize_max_tokens_keys(payload, model)

    return payload, False
