"""
XAI Responses API Provider — native stateful provider for xAI Grok models.

Uses /v1/responses (REST) instead of /v1/chat/completions.
Key win: previous_response_id means turn N+1 only sends the new user message —
full context (system prompt + history) is cached on xAI servers for 30 days.

Cost model: billed for full context but cached tokens are cheaper.
Network payload: dramatically reduced after turn 1.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class XAIResponsesProvider(LLMProvider):
    """
    Native xAI Responses API provider.

    Drop-in replacement for CustomProvider when using xAI.
    Adds stateful continuation via previous_response_id — the main agent loop
    threads this through so each turn only sends the new user message.
    """

    API_BASE = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str,
        default_model: str = "grok-4-1-fast",
        store_messages: bool = True,
    ) -> None:
        super().__init__(api_key=api_key, api_base=self.API_BASE)
        self.default_model = default_model
        self.store_messages = store_messages
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public interface (matches LLMProvider.chat signature + previous_response_id)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        previous_response_id: str | None = None,
    ) -> LLMResponse:
        """
        Send a request to /v1/responses.

        If previous_response_id is provided:
        - AND messages contain a system prompt (first call in a turn):
            → server has history; only send the last user message.
        - AND messages are only tool results (continuation within a turn):
            → server has the assistant tool-call message; only send tool results.

        If no previous_response_id: send full context (first turn ever, or after expiry).
        """
        is_first_call = any(m.get("role") == "system" for m in messages)
        has_tool_results = any(m.get("role") == "tool" for m in messages)

        if previous_response_id and is_first_call and not has_tool_results:
            # Cross-turn continuation: server has everything up to last response.
            # Only send the new user message.
            input_items = [
                self._translate_message(m)
                for m in messages
                if m.get("role") == "user"
            ][-1:]
        elif previous_response_id and has_tool_results and not is_first_call:
            # Within-turn tool continuation: server has assistant tool-call message.
            # Only send tool result items.
            input_items = [
                self._translate_message(m)
                for m in messages
                if m.get("role") == "tool"
            ]
        else:
            # Full context: first turn ever, or fallback after expiry.
            input_items = [
                self._translate_message(m)
                for m in messages
                if self._keep_message(m)
            ]

        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "input": input_items,
            "store": self.store_messages,
            "max_output_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if tools:
            payload["tools"] = self._translate_tools(tools)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{self.API_BASE}/responses",
                    headers=self._headers,
                    json=payload,
                )
                r.raise_for_status()
                return self._parse(r.json())
        except httpx.HTTPStatusError as e:
            body = e.response.text[:400]
            # Graceful degradation: if previous_response_id expired (404/400),
            # retry with full context.
            if e.response.status_code in (400, 404) and previous_response_id:
                return await self.chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    previous_response_id=None,  # full rebuild
                )
            return LLMResponse(content=f"xAI API error {e.response.status_code}: {body}", finish_reason="error")
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        """Parse /v1/responses response into LLMResponse."""
        output = data.get("output", [])
        content: str | None = None
        tool_calls: list[ToolCallRequest] = []

        for item in output:
            item_type = item.get("type")

            if item_type == "message":
                # Extract text from content array
                parts = item.get("content", [])
                texts = [
                    p.get("text", "")
                    for p in parts
                    if isinstance(p, dict) and p.get("type") == "output_text"
                ]
                content = "\n".join(texts).strip() or None

            elif item_type == "function_call":
                raw_args = item.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, ValueError):
                    args = {}
                tool_calls.append(ToolCallRequest(
                    id=item.get("call_id", item.get("id", "")),
                    name=item.get("name", ""),
                    arguments=args,
                ))

        # Determine finish reason
        if tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = data.get("stop_reason", "stop")

        # Usage — map xAI keys to standard keys
        raw_usage = data.get("usage", {})
        usage: dict[str, int] = {}
        if raw_usage:
            usage = {
                "prompt_tokens": raw_usage.get("input_tokens", 0),
                "completion_tokens": raw_usage.get("output_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            response_id=data.get("id"),
        )

    # ------------------------------------------------------------------
    # Message translation: Chat Completions format → Responses API input
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_message(msg: dict[str, Any]) -> bool:
        """Filter out messages that don't translate cleanly (e.g. bare assistant w/ only tool_calls)."""
        role = msg.get("role")
        # Keep system, user, tool results always.
        if role in ("system", "user", "tool"):
            return True
        # Keep assistant messages that have actual text content.
        if role == "assistant":
            content = msg.get("content")
            # If only tool_calls and no text, skip — server already has it when stateful.
            # Include if there's text content worth preserving.
            return bool(content)
        return False

    @staticmethod
    def _translate_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Translate a single Chat Completions message to Responses API input format."""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Normalise content: handle list format (vision, multi-part)
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
            ]
            content = " ".join(text_parts).strip() or "(media)"

        # Tool results → function_call_output
        if role == "tool":
            return {
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content or "",
            }

        # System / user / assistant → standard role message
        return {
            "role": role,
            "content": content or "",
        }

    @staticmethod
    def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Translate Chat Completions tool format to Responses API tool format.

        Chat Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Responses API:    {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        translated = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function", {})
            translated.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        return translated
