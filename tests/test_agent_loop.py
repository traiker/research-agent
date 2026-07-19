from typing import List
from unittest.mock import MagicMock

import pytest

from agent.loop import AgentLoop
from agent.providers.base import AgentProvider, ModelTurn, ToolCallRequest
from harness.executor import ToolExecutionResult


class FakeProvider(AgentProvider):
    """Test-Double: liefert eine vorgegebene Sequenz von ModelTurns, ohne
    eine echte API anzusprechen - prüft, dass AgentLoop providerunabhängig
    korrekt funktioniert."""

    def __init__(self, turns: List[ModelTurn]) -> None:
        super().__init__(model="fake-model", tools_schema=[])
        self._turns = list(turns)
        self.started_with = None
        self.submitted_results: List[List[ToolExecutionResult]] = []

    def start(self, user_message: str) -> None:
        self.started_with = user_message

    def call(self) -> ModelTurn:
        return self._turns.pop(0)

    def submit_tool_results(self, results: List[ToolExecutionResult]) -> None:
        self.submitted_results.append(results)


def make_executor_stub(result_content: str = "tool-ok") -> MagicMock:
    executor = MagicMock()
    executor.execute.return_value = ToolExecutionResult("call_1", result_content)
    return executor


def test_run_returns_text_immediately_when_no_tool_use():
    provider = FakeProvider([ModelTurn(text="Antwort", tool_calls=[], stop_reason="end_turn")])
    executor = make_executor_stub()

    loop = AgentLoop(provider=provider, executor=executor, max_iterations=10)
    answer = loop.run("Frage")

    assert answer == "Antwort"
    assert provider.started_with == "Frage"
    executor.execute.assert_not_called()


def test_run_executes_tool_calls_and_feeds_results_back():
    provider = FakeProvider(
        [
            ModelTurn(
                text="",
                tool_calls=[ToolCallRequest(id="call_1", name="web_search", input={"query": "x"})],
                stop_reason="tool_use",
            ),
            ModelTurn(text="Fertig", tool_calls=[], stop_reason="end_turn"),
        ]
    )
    executor = make_executor_stub()

    loop = AgentLoop(provider=provider, executor=executor, max_iterations=10)
    answer = loop.run("Frage")

    assert answer == "Fertig"
    executor.execute.assert_called_once_with("call_1", "web_search", {"query": "x"})
    assert len(provider.submitted_results) == 1
    assert provider.submitted_results[0][0].content == "tool-ok"


def test_run_raises_after_max_iterations_without_final_answer():
    # Jede Iteration fordert erneut einen Tool-Call an - nie eine finale Antwort.
    turns = [
        ModelTurn(
            text="",
            tool_calls=[ToolCallRequest(id=f"call_{i}", name="web_search", input={})],
            stop_reason="tool_use",
        )
        for i in range(3)
    ]
    provider = FakeProvider(turns)
    executor = make_executor_stub()

    loop = AgentLoop(provider=provider, executor=executor, max_iterations=3)

    with pytest.raises(RuntimeError):
        loop.run("Frage")
