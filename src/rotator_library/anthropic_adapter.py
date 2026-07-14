# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
from .config.defaults import TRACE

from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .anthropic_compat.models import AnthropicMessagesRequest, AnthropicCountTokensRequest

from .utils.json_utils import json_dumps_str

lib_logger = logging.getLogger("rotator_library")


class AnthropicAdapter:
    def __init__(self, acompletion_fn, token_count_fn, extract_provider_fn, all_credentials, enable_request_logging):
        self._acompletion = acompletion_fn
        self._token_count = token_count_fn
        self._extract_provider_from_model = extract_provider_fn
        self.all_credentials = all_credentials
        self.enable_request_logging = enable_request_logging

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        raw_body_data: Optional[dict] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        This method accepts requests in Anthropic's format, translates them to
        OpenAI format internally, processes them through the existing acompletion
        method, and returns responses in Anthropic's format.

        Args:
            request: An AnthropicMessagesRequest object
            raw_request: Optional raw request object for disconnect checks
            pre_request_callback: Optional async callback before each API request

        Returns:
            For non-streaming: dict in Anthropic Messages format
            For streaming: AsyncGenerator yielding Anthropic SSE format strings
        """
        from .anthropic_compat import (
            translate_anthropic_request,
            openai_to_anthropic_response,
            anthropic_streaming_wrapper,
            build_tool_schema_map,
            TranslationAuditLog,
        )
        from .token_calculator import count_input_tokens_result
        import uuid

        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        original_model = request.model

        # Extract provider from model for logging
        provider = self._extract_provider_from_model(original_model) or "unknown"

        # Create Anthropic transaction logger if request logging is enabled
        anthropic_logger = None
        if self.enable_request_logging:
            from .transaction_logger import TransactionLogger

            anthropic_logger = TransactionLogger(
                provider,
                original_model,
                enabled=True,
                api_format="ant",
            )
            # Log original Anthropic request
            await anthropic_logger.log_request(
                raw_body_data if raw_body_data is not None else request.model_dump(exclude_none=True),
                filename="anthropic_request.json",
            )

        # Translate Anthropic request to OpenAI format
        openai_request = translate_anthropic_request(request)

        # Emit translation audit log
        TranslationAuditLog.log_from_metadata(
            request_id=request_id,
            direction="anthropic_to_openai",
            translation_metadata=openai_request,
        )

        # Strip internal metadata keys before forwarding to provider
        # _cache_control_metadata is preserved in messages for round-trip fidelity
        openai_request.pop("_translation_audit", None)

        tool_schemas = build_tool_schema_map(openai_request.get("tools"))

        # Pass parent log directory to acompletion for nested logging
        if anthropic_logger and anthropic_logger.log_dir:
            openai_request["_parent_log_dir"] = anthropic_logger.log_dir

        # raw_request is not passed to LiteLLM - it may contain client headers

        if request.stream:
            # Streaming response
            # [FIX] Don't pass raw_request to LiteLLM - it may contain client headers
            # (x-api-key, anthropic-version, etc.) that shouldn't be forwarded to providers

            # Pre-compute input tokens for fallback when provider doesn't return usage
            # This is critical for Claude Code's context management to work correctly
            # with providers like Kilocode that don't support stream_options
            precomputed_input_tokens = None
            try:
                messages = openai_request.get("messages", [])
                tools = openai_request.get("tools")
                tool_choice = openai_request.get("tool_choice")
                if messages:
                    token_count = count_input_tokens_result(
                        messages=messages,
                        model=original_model,
                        tools=tools,
                        tool_choice=tool_choice,
                        allow_timed_exact=True,
                    )
                    precomputed_input_tokens = token_count.count
                    lib_logger.log(
                        TRACE,
                        "Pre-computed input tokens for "
                        f"{original_model}: {precomputed_input_tokens} "
                        f"(exact={token_count.exact}, reason={token_count.reason})"
                    )
            except (ValueError, TypeError, Exception) as e:
                lib_logger.warning(f"Failed to pre-compute input tokens: {e}")

            response_generator = self._acompletion(
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Create disconnect checker if raw_request provided
            is_disconnected = None
            if raw_request is not None and hasattr(raw_request, "is_disconnected"):
                is_disconnected = raw_request.is_disconnected

            # Extract cache_control metadata from messages for round-trip fidelity
            cache_control_map = {}
            for msg in openai_request.get("messages", []):
                if isinstance(msg, dict) and "_cache_control_metadata" in msg:
                    cache_control_map.update(msg["_cache_control_metadata"])

            # Return the streaming wrapper
            # Note: For streaming, the anthropic response logging happens in the wrapper
            return anthropic_streaming_wrapper(
                openai_stream=response_generator,
                original_model=original_model,
                request_id=request_id,
                is_disconnected=is_disconnected,
                transaction_logger=anthropic_logger,
                precomputed_input_tokens=precomputed_input_tokens,
                cache_control_map=cache_control_map if cache_control_map else None,
                tool_schemas=tool_schemas,
            )
        else:
            # Non-streaming response
            # [FIX] Don't pass raw_request to LiteLLM - it may contain client headers
            # (x-api-key, anthropic-version, etc.) that shouldn't be forwarded to providers
            response = await self._acompletion(
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Convert OpenAI response to Anthropic format
            openai_response = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )
            anthropic_response = openai_to_anthropic_response(
                openai_response,
                original_model,
                tool_schemas=tool_schemas,
            )

            # Override the ID with our request ID
            anthropic_response["id"] = request_id

            # Log Anthropic response
            if anthropic_logger:
                await anthropic_logger.log_response(
                    anthropic_response,
                    filename="anthropic_response.json",
                )

            return anthropic_response

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        """
        Handle Anthropic count_tokens API requests.

        Counts the number of tokens that would be used by a Messages API request.
        This is useful for estimating costs and managing context windows.

        Args:
            request: An AnthropicCountTokensRequest object

        Returns:
            Dict with input_tokens count in Anthropic format
        """
        from .anthropic_compat import (
            anthropic_to_openai_messages,
            anthropic_to_openai_tools,
        )

        anthropic_request = request.model_dump(exclude_none=True)

        openai_messages = anthropic_to_openai_messages(
            anthropic_request.get("messages", []), anthropic_request.get("system")
        )

        # Count tokens for messages
        message_tokens = self._token_count(
            model=request.model,
            messages=openai_messages,
        )

        # Count tokens for tools if present
        tool_tokens = 0
        if request.tools:
            # Tools add tokens based on their definitions
            # Convert to JSON string and count tokens for tool definitions
            openai_tools = anthropic_to_openai_tools(
                [tool.model_dump() for tool in request.tools]
            )
            if openai_tools:
                # Serialize tools to count their token contribution
                tools_text = json_dumps_str(openai_tools)
                tool_tokens = self._token_count(
                    model=request.model,
                    text=tools_text,
                )

        total_tokens = message_tokens + tool_tokens

        return {"input_tokens": total_tokens}
