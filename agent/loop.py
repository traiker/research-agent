"""Agent-Loop: ruft den konfigurierten Provider (Claude oder OpenAI) auf,
spielt Tool-Ergebnisse über den Executor zurück, bis das Modell eine
finale Antwort ohne weitere Tool-Calls liefert. Max. 10 Iterationen als
Sicherheitsgrenze - providerunabhängig, da hier zentral durchgesetzt."""

from __future__ import annotations

from agent.providers.base import AgentProvider
from harness.executor import ToolExecutor

MAX_ITERATIONS = 10


class AgentLoop:
    def __init__(
        self,
        provider: AgentProvider,
        executor: ToolExecutor,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.provider = provider
        self.executor = executor
        self.max_iterations = max_iterations

    def run(self, user_message: str) -> str:
        self.provider.start(user_message)

        for _ in range(self.max_iterations):
            turn = self.provider.call()

            if not turn.wants_tool_use:
                return turn.text

            results = [
                self.executor.execute(call.id, call.name, call.input)
                for call in turn.tool_calls
            ]
            self.provider.submit_tool_results(results)

        raise RuntimeError(
            f"Agent hat innerhalb von {self.max_iterations} Iterationen keine "
            "finale Antwort geliefert (Sicherheitsgrenze erreicht)."
        )
