"""Beispiel-Tool: einfache Web-Suche über die DuckDuckGo
Instant-Answer-API (kein API-Key nötig, daher für ein Beispielprojekt
geeignet - für produktiven Einsatz durch eine echte Such-API ersetzen)."""

from __future__ import annotations

from typing import Any, Dict

import requests

from harness.executor import TransientToolError

SEARCH_ENDPOINT = "https://api.duckduckgo.com/"

DESCRIPTION = (
    "Durchsucht das Web über die DuckDuckGo Instant-Answer-API und liefert "
    "eine kurze Zusammenfassung sowie verwandte Themen zu einer Suchanfrage."
)

INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Suchanfrage, z.B. 'Anthropic Claude Modelle'",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximale Anzahl an verwandten Themen (Default 3)",
        },
    },
    "required": ["query"],
}


def handler(tool_input: Dict[str, Any]) -> str:
    query = tool_input["query"]
    max_results = int(tool_input.get("max_results") or 3)

    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}

    try:
        response = requests.get(SEARCH_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise TransientToolError(f"web_search: Timeout bei der Anfrage: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise TransientToolError(f"web_search: Verbindungsfehler: {exc}") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status is not None and 500 <= status < 600:
            raise TransientToolError(f"web_search: Serverfehler ({status})") from exc
        raise

    data = response.json()

    lines = []
    if data.get("AbstractText"):
        lines.append(data["AbstractText"])

    for topic in data.get("RelatedTopics", []):
        if len(lines) >= max_results + 1:
            break
        text = topic.get("Text") if isinstance(topic, dict) else None
        if text:
            lines.append(f"- {text}")

    if not lines:
        return f"Keine Ergebnisse für '{query}' gefunden."

    return "\n".join(lines)
