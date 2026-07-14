# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Tiered Context Compaction — deterministic three-phase message compaction.

Before sending a request to a provider, if total input tokens exceed the
model's context window threshold, apply deterministic compaction:

Phase 1: Drop system messages containing the marker "[proxy-nudge]"
Phase 2: Truncate old tool messages, preserving tool_call_id
Phase 3: Drop old assistant text, preserving only tool_calls

If still over limit after all phases, raise ContextOverflowError.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .error_types import ContextOverflowError

logger = logging.getLogger("rotator_library")

NUDGE_MARKER = "[proxy-nudge]"


@dataclass
class CompactionConfig:
    """Configuration for context compaction behavior."""

    enabled: bool = False
    threshold: float = 0.90  # 90% of context window
    keep_recent_tools: int = 5
    keep_recent_assistant: int = 3
    preserve_system: bool = True  # preserve non-nudge system messages


class ContextCompactor:
    """Deterministic three-phase message compactor.

    Operates on a deep copy of request_data — never mutates the original.
    """

    def __init__(
        self,
        config: CompactionConfig,
        token_counter: Callable[[list, str], int],
    ):
        """
        Args:
            config: CompactionConfig with thresholds and tunables.
            token_counter: Callable(messages, model) -> int that returns
                the token count for a list of messages.
        """
        self.config = config
        self._token_counter = token_counter

    def _count_tokens(self, messages: list, model: str) -> int:
        """Count tokens via injected callable."""
        return self._token_counter(messages, model)

    def compact(
        self,
        request_data: dict,
        *,
        context_window: int,
        model: str,
    ) -> dict:
        """Return request_data unchanged or a copy with compacted messages.

        Args:
            request_data: Original request payload (never mutated).
            context_window: Model's context window in tokens.
            model: Model identifier for token counting.

        Returns:
            Original request_data when no compaction is needed, otherwise a
            deep copy with messages compacted to fit.

        Raises:
            ContextOverflowError: If messages still exceed threshold after
                all compaction phases.
        """
        messages: list = request_data.get("messages", [])
        if not messages:
            return request_data

        token_limit = int(context_window * self.config.threshold)
        current_tokens = self._count_tokens(messages, model)

        logger.debug(
            "Context compaction start: model=%s tokens=%d limit=%d (%.0f%% of %d)",
            model, current_tokens, token_limit,
            self.config.threshold * 100, context_window,
        )

        if current_tokens <= token_limit:
            logger.debug("Context compaction: no compaction needed")
            return request_data

        # Deep copy only when compacting; caller data is never mutated.
        result = copy.deepcopy(request_data)
        messages = result.get("messages", [])

        # --- Phase 1: Drop nudge system messages ---
        messages = self._phase1_drop_nudges(messages)
        result["messages"] = messages
        current_tokens = self._count_tokens(messages, model)
        logger.debug(
            "Context compaction phase1 (drop nudges): tokens=%d", current_tokens,
        )
        if current_tokens <= token_limit:
            return result

        # --- Phase 2: Truncate old tool messages ---
        messages = self._phase2_truncate_old_tools(messages)
        result["messages"] = messages
        current_tokens = self._count_tokens(messages, model)
        logger.debug(
            "Context compaction phase2 (truncate tools): tokens=%d", current_tokens,
        )
        if current_tokens <= token_limit:
            return result

        # --- Phase 3: Drop old assistant text ---
        messages = self._phase3_drop_old_assistant_text(messages)
        result["messages"] = messages
        current_tokens = self._count_tokens(messages, model)
        logger.debug(
            "Context compaction phase3 (drop assistant text): tokens=%d", current_tokens,
        )
        if current_tokens <= token_limit:
            return result

        # --- Emergency cutoff ---
        tokens_over = current_tokens - token_limit
        logger.error(
            "Context compaction emergency cutoff: model=%s tokens=%d limit=%d over=%d",
            model, current_tokens, token_limit, tokens_over,
        )
        raise ContextOverflowError(
            model=model,
            tokens_over_limit=tokens_over,
            current_tokens=current_tokens,
            context_limit=token_limit,
        )

    # ---- Phase implementations ----

    @staticmethod
    def _phase1_drop_nudges(messages: list) -> list:
        """Drop system messages where content is exactly [proxy-nudge] marker."""
        kept: list = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system" and isinstance(content, str) and content.strip() == NUDGE_MARKER:
                logger.debug("Phase1: dropping nudge system message")
                continue
            kept.append(msg)
        return kept

    def _phase2_truncate_old_tools(self, messages: list) -> list:
        """Truncate tool messages older than keep_recent_tools turns.

        A 'turn' here is counted from the end of the messages list.
        Tool messages beyond the cutoff have their content replaced with
        a truncation notice, but tool_call_id is preserved.
        """
        keep = self.config.keep_recent_tools
        if keep < 0:
            return messages

        # Identify tool message indices and count from the end
        tool_indices: List[int] = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "tool":
                tool_indices.append(i)

        # tool_indices is newest-first; skip the first `keep` ones
        to_truncate = tool_indices[keep:]

        if not to_truncate:
            return messages

        result = list(messages)
        for idx in to_truncate:
            original = result[idx]
            content = original.get("content", "")
            content_str = content if isinstance(content, str) else str(content)
            # Rough token estimate for the content being dropped
            approx_tokens = len(content_str.split()) * 4 // 3
            truncated = {
                "role": "tool",
                "content": f"[Result truncated: {approx_tokens} tokens]",
            }
            # Preserve tool_call_id if present
            if "tool_call_id" in original:
                truncated["tool_call_id"] = original["tool_call_id"]
            # Preserve name if present
            if "name" in original:
                truncated["name"] = original["name"]
            result[idx] = truncated
        return result

    def _phase3_drop_old_assistant_text(self, messages: list) -> list:
        """Drop content from old assistant messages, preserving tool_calls.

        Assistant messages older than keep_recent_assistant turns have
        their text content removed; only tool_calls arrays are kept.
        """
        keep = self.config.keep_recent_assistant
        if keep < 0:
            return messages

        # Identify assistant message indices, newest-first
        assistant_indices: List[int] = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                assistant_indices.append(i)

        # Skip the first `keep` ones (most recent)
        to_strip = assistant_indices[keep:]

        if not to_strip:
            return messages

        result = list(messages)
        for idx in to_strip:
            original = result[idx]
            stripped: dict = {"role": "assistant"}
            # Preserve tool_calls
            if "tool_calls" in original and original["tool_calls"]:
                stripped["tool_calls"] = original["tool_calls"]
                stripped["content"] = None
            else:
                # No tool_calls — keep content but it's old assistant text
                # We still strip it to save tokens
                stripped["content"] = ""
            result[idx] = stripped
        return result
