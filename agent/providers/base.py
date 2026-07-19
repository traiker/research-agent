"""Providerabstraktion: kapselt alles API-Spezifische (Message-Wire-Format,
Tool-Schema-Übersetzung, der eigentliche HTTP-Call), damit der Agent-Loop
sowie die komplette Tool-Harness (Registry, Sandbox, Executor) unverändert
mit jedem Provider funktionieren.

Jeder Provider hält seine Konversationshistorie intern im eigenen,
providerspezifischen Format (Claude: content-blocks + separates `system`;
OpenAI: flache Message-Liste mit `role: "tool"`-Einträgen). Nach außen zeigt
er nur die providerneutralen Typen unten."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harness.executor import ToolExecutionResult


@dataclass
class ToolCallRequest:
    """Providerneutrale Sicht auf einen vom Modell angeforderten Tool-Call."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ModelTurn:
    """Providerneutrale Sicht auf eine einzelne Modell-Antwort."""

    text: str
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def wants_tool_use(self) -> bool:
        return bool(self.tool_calls)


class AgentProvider(ABC):
    """Ein Provider übersetzt zwischen dem Agent-Loop (providerneutral) und
    einer konkreten LLM-API. Tool-Schemas kommen providerneutral von
    `ToolRegistry.to_api_tools()` (Liste aus {name, description, input_schema})
    herein - jeder Provider konvertiert sie intern in sein eigenes Format."""

    def __init__(
        self,
        model: str,
        tools_schema: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tools_schema = tools_schema
        self.system = system
        self.messages: List[Any] = []  # providereigenes Wire-Format

    @abstractmethod
    def start(self, user_message: str) -> None:
        """Initialisiert die Historie mit der ersten User-Nachricht."""

    @abstractmethod
    def call(self) -> ModelTurn:
        """Ruft das Modell mit der aktuellen Historie auf, hängt die
        Assistant-Antwort providerspezifisch an die Historie an und gibt
        eine providerneutrale ModelTurn zurück."""

    @abstractmethod
    def submit_tool_results(self, results: List[ToolExecutionResult]) -> None:
        """Hängt die Tool-Ergebnisse providerspezifisch formatiert an die
        Historie an, bevor die nächste Iteration `call()` aufruft."""
