"""Beispiel-Tool: ruft den Textinhalt einer URL ab. Nutzt die Sandbox für
die Domain-Allowlist und die maximale Antwortgröße - das Timeout pro Call
übernimmt der Executor über sandbox.run()."""

from __future__ import annotations

from typing import Any, Dict

import requests

from harness.executor import TransientToolError
from harness.registry import ToolHandler
from harness.sandbox import Sandbox

DESCRIPTION = (
    "Ruft den Textinhalt einer URL ab. Nur Domains aus der Sandbox-Allowlist "
    "sind erlaubt; die Antwort wird auf die konfigurierte Maximalgröße gekürzt."
)

INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Vollständige URL, die abgerufen werden soll (inkl. https://)",
        }
    },
    "required": ["url"],
}


def make_handler(sandbox: Sandbox) -> ToolHandler:
    """Baut den web_fetch-Handler; bindet ihn an eine konkrete Sandbox-Instanz,
    damit Domain-Allowlist und Größenlimit dort zentral konfiguriert bleiben."""

    def handler(tool_input: Dict[str, Any]) -> str:
        url = tool_input["url"]

        # Wirft SandboxViolation, falls die Domain nicht erlaubt ist - der
        # Executor fängt das ab und retried es (bewusst) nicht.
        sandbox.check_domain_allowed(url)

        try:
            response = requests.get(
                url, timeout=10, headers={"User-Agent": "research-agent-example/1.0"}
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise TransientToolError(f"web_fetch: Timeout bei der Anfrage: {exc}") from exc
        except requests.exceptions.ConnectionError as exc:
            raise TransientToolError(f"web_fetch: Verbindungsfehler: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 500 <= status < 600:
                raise TransientToolError(f"web_fetch: Serverfehler ({status})") from exc
            raise

        return sandbox.enforce_max_size(response.text)

    return handler
