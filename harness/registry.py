"""Zentrale Tool-Registry: bündelt Schema + Handler und generiert die
`tools`-Liste für die Anthropic Messages API."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List

# Ein Handler bekommt das geparste `input`-Dict des tool_use-Blocks und liefert
# einen String zurück, der als tool_result-Content an die API zurückgeht.
ToolHandler = Callable[[Dict[str, Any]], str]


@dataclasses.dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler
    timeout_seconds: float = 15.0


class ToolRegistry:
    """Hält alle registrierten Tools und erzeugt daraus die API-Tool-Liste."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' ist bereits registriert")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unbekanntes Tool: {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def to_api_tools(self) -> List[Dict[str, Any]]:
        """Baut die `tools`-Liste für `client.messages.create(...)`."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]
