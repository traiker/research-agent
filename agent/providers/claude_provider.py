"""Claude-Provider: dünner Wrapper um `client.messages.create(...)`.

Die Tool-Schemas aus der Registry ({name, description, input_schema}) sind
bereits im Format, das die Anthropic Messages API erwartet - keine
Übersetzung nötig."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import anthropic

from agent.providers.base import AgentProvider, ModelTurn, ToolCallRequest
from harness.executor import ToolExecutionResult

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeProvider(AgentProvider):
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        tools_schema: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, tools_schema=tools_schema, system=system)
        self.client = client
        self.max_tokens = max_tokens

    def start(self, user_message: str) -> None:
        self.messages = [{"role": "user", "content": user_message}]

    def call(self) -> ModelTurn:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=self.tools_schema,
            thinking={"type": "adaptive"},
            messages=self.messages,
        )
        if self.system:
            kwargs["system"] = self.system

        response = self.client.messages.create(**kwargs)
        self.messages.append({"role": "assistant", "content": response.content})

        text_parts = [block.text for block in response.content if block.type == "text"]
        tool_calls = [
            ToolCallRequest(id=block.id, name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        return ModelTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )

    def submit_tool_results(self, results: List[ToolExecutionResult]) -> None:
        self.messages.append({"role": "user", "content": [r.to_api_block() for r in results]})
