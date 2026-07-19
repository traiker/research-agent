"""Agent-Loop: ruft die Anthropic Messages API mit Tool-Use auf, spielt
Tool-Ergebnisse zurück, bis Claude eine finale Antwort ohne weitere
Tool-Calls liefert. Max. 10 Iterationen als Sicherheitsgrenze."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import anthropic

from harness.executor import ToolExecutor

DEFAULT_MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 10


class AgentLoop:
    def __init__(
        self,
        client: anthropic.Anthropic,
        executor: ToolExecutor,
        tools: List[Dict[str, Any]],
        model: str = DEFAULT_MODEL,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.client = client
        self.executor = executor
        self.tools = tools
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations

    def run(self, user_message: str) -> str:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message}]

        for _ in range(self.max_iterations):
            response = self._call_model(messages)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return self._final_text(response)

            tool_results = [
                self.executor.execute(block.id, block.name, block.input).to_api_block()
                for block in response.content
                if block.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(
            f"Agent hat innerhalb von {self.max_iterations} Iterationen keine "
            "finale Antwort geliefert (Sicherheitsgrenze erreicht)."
        )

    def _call_model(self, messages: List[Dict[str, Any]]) -> Any:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=self.tools,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        if self.system:
            kwargs["system"] = self.system
        return self.client.messages.create(**kwargs)

    @staticmethod
    def _final_text(response: Any) -> str:
        parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(parts)
