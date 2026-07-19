"""OpenAI-Provider: übersetzt die providerneutralen Tool-Schemas in das
OpenAI-Function-Calling-Format (`{"type": "function", "function": {...}}`)
und die Chat-Completions-Response zurück in ModelTurn/ToolCallRequest.

Hinweis: `max_tokens` ist bei den neueren Reasoning-Modellen (o1/o3/...)
durch `max_completion_tokens` ersetzt worden. Für die Standard-Chat-Modelle
(gpt-4o, gpt-4.1, ...) funktioniert `max_tokens` weiterhin. Falls du ein
Reasoning-Modell nutzen willst, hier entsprechend anpassen."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agent.providers.base import AgentProvider, ModelTurn, ToolCallRequest
from harness.executor import ToolExecutionResult

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(AgentProvider):
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools_schema: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, tools_schema=tools_schema, system=system)
        self.client = client
        self.max_tokens = max_tokens
        self._openai_tools = [self._to_openai_tool(t) for t in tools_schema]

    @staticmethod
    def _to_openai_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }

    def start(self, user_message: str) -> None:
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})
        self.messages.append({"role": "user", "content": user_message})

    def call(self) -> ModelTurn:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
            tools=self._openai_tools or None,
        )
        message = response.choices[0].message

        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        self.messages.append(assistant_entry)

        tool_calls: List[ToolCallRequest] = []
        for tc in message.tool_calls or []:
            try:
                tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                tool_input = {}
            tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, input=tool_input))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return ModelTurn(text=message.content or "", tool_calls=tool_calls, stop_reason=stop_reason)

    def submit_tool_results(self, results: List[ToolExecutionResult]) -> None:
        for result in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_use_id,
                    "content": result.content,
                }
            )
