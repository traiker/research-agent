"""Führt tool_use-Blöcke gegen die Registry aus: fängt Exceptions ab, gibt
strukturierte Fehler statt Crashes zurück und retried transiente Fehler
(max. 2 Retries, exponentielles Backoff)."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from harness.logging import ToolCallLogger
from harness.registry import ToolRegistry
from harness.sandbox import Sandbox, SandboxViolation, ToolTimeoutError


class TransientToolError(Exception):
    """Von Tool-Handlern geworfen, um einen vorübergehenden (retrybaren)
    Fehler zu signalisieren (z.B. Netzwerk-Timeout, 5xx-Serverfehler)."""


@dataclass
class ToolExecutionResult:
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_api_block(self) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        return block


class ToolExecutor:
    """Führt registrierte Tools über die Sandbox aus und liefert für jeden
    tool_use-Block garantiert ein ToolExecutionResult - nie eine Exception."""

    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: Sandbox,
        logger: ToolCallLogger,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.5,
    ) -> None:
        self.registry = registry
        self.sandbox = sandbox
        self.logger = logger
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

    def execute(
        self, tool_use_id: str, tool_name: str, tool_input: Dict[str, Any]
    ) -> ToolExecutionResult:
        try:
            tool = self.registry.get(tool_name)
        except KeyError as exc:
            self.logger.log(tool_name, tool_input, error=str(exc), duration_ms=0.0)
            return ToolExecutionResult(tool_use_id, f"Error: {exc}", is_error=True)

        last_error: Optional[Exception] = None
        attempt = 0

        while attempt <= self.max_retries:
            start = time.monotonic()
            try:
                result = self.sandbox.run(
                    tool.handler, tool_input, timeout_seconds=tool.timeout_seconds
                )
                duration_ms = (time.monotonic() - start) * 1000
                self.logger.log(
                    tool_name, tool_input, output=result, duration_ms=duration_ms,
                    attempt=attempt,
                )
                return ToolExecutionResult(tool_use_id, str(result))

            except (TransientToolError, ToolTimeoutError) as exc:
                # Transiente Fehler: mit exponentiellem Backoff erneut versuchen.
                last_error = exc
                duration_ms = (time.monotonic() - start) * 1000
                self.logger.log(
                    tool_name, tool_input, error=str(exc), duration_ms=duration_ms,
                    attempt=attempt, retryable=True,
                )
                if attempt == self.max_retries:
                    break
                backoff = self.base_backoff_seconds * (2**attempt) + random.uniform(0, 0.1)
                time.sleep(backoff)
                attempt += 1

            except SandboxViolation as exc:
                # Policy-Verletzungen (z.B. Domain nicht erlaubt) sind nicht
                # transient - kein Retry, sofort als Fehler zurückgeben.
                duration_ms = (time.monotonic() - start) * 1000
                self.logger.log(
                    tool_name, tool_input, error=str(exc), duration_ms=duration_ms,
                    attempt=attempt, retryable=False,
                )
                return ToolExecutionResult(tool_use_id, f"Error: {exc}", is_error=True)

            except Exception as exc:  # noqa: BLE001 - Tool-Handler sind nicht vertrauenswürdig
                # Alles andere (Bugs in Handlern, unerwartete Exceptions) fängt
                # der Executor ab, statt den Agent-Loop crashen zu lassen.
                duration_ms = (time.monotonic() - start) * 1000
                self.logger.log(
                    tool_name, tool_input, error=str(exc), duration_ms=duration_ms,
                    attempt=attempt, retryable=False,
                )
                return ToolExecutionResult(tool_use_id, f"Error: {exc}", is_error=True)

        return ToolExecutionResult(tool_use_id, f"Error: {last_error}", is_error=True)
