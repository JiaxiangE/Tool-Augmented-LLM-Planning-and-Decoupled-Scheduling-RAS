"""
LLM Backend Abstraction Layer.

Provides a unified interface for calling LLMs with tool/function-calling support.
Only QwenBackend is fully implemented. OpenAI and OpenRouter
backends are scaffolding stubs for future multi-provider comparisons.
"""
from __future__ import annotations

import json
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """
    Abstract base for LLM providers.

    All backends normalize responses into a unified format:
      - thought: str (the model's reasoning text)
      - tool_call: dict | None (the tool invocation, if any)

    tool_call format (when present):
      {"name": "tool_name", "arguments": {...}}
    """

    @abstractmethod
    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Send messages + tool specs to the LLM and get back a response.

        Args:
            messages: OpenAI-format message list (role, content, etc.)
            tools: OpenAI-format tool/function specs

        Returns:
            (thought, tool_call_or_None)
        """
        ...

    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total token count for a message list."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


# ---------------------------------------------------------------------------
# Qwen Backend
# ---------------------------------------------------------------------------

class QwenBackend(LLMBackend):
    """
    Qwen backend via DashScope OpenAI-compatible API.

    Uses the openai library with a custom base_url pointing to DashScope.
    Set environment variable DASHSCOPE_API_KEY to your API key.
    """

    DEFAULT_MODEL = "qwen3-max"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Lazy-init the client
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for QwenBackend. "
                    "Install it with: pip install openai"
                )
            if not self._api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. Please set it as an environment variable "
                    "or pass api_key to QwenBackend constructor.\n"
                    "  export DASHSCOPE_API_KEY='your-key-here'  (Linux/Mac)\n"
                    "  $env:DASHSCOPE_API_KEY='your-key-here'    (PowerShell)"
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("Calling %s with %d messages, %d tools", self._model, len(messages), len(tools))

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Extract thought (content text)
        thought = message.content or ""

        # Extract tool call if present
        tool_call = None
        if message.tool_calls and len(message.tool_calls) > 0:
            tc = message.tool_calls[0]
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                arguments = {}
            tool_call = {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": arguments,
            }

        logger.debug("Response: thought=%d chars, tool_call=%s", len(thought), tool_call["name"] if tool_call else "None")
        return thought, tool_call

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Rough token estimate for Qwen.

        Uses ~4 chars per token heuristic. For precise counting,
        use tiktoken with cl100k_base encoding.
        """
        total_chars = sum(
            len(json.dumps(m, ensure_ascii=False)) for m in messages
        )
        return total_chars // 4


# ---------------------------------------------------------------------------
# OpenAI Backend (stub for future multi-provider comparison)
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend. Not implemented."""

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError("OpenAIBackend is a scaffolding stub")

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        raise NotImplementedError("OpenAIBackend is a scaffolding stub")


# ---------------------------------------------------------------------------
# OpenRouter Backend (stub)
# ---------------------------------------------------------------------------

class OpenRouterBackend(LLMBackend):
    """OpenRouter multi-model proxy backend. Not implemented."""

    def __init__(self, model: str = "openai/gpt-4o", **kwargs):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError("OpenRouterBackend is a scaffolding stub")

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        raise NotImplementedError("OpenRouterBackend is a scaffolding stub")
