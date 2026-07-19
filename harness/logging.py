"""Strukturiertes JSON-Logging für jeden Tool-Call: timestamp, tool_name,
input, output/error und duration_ms - eine Zeile pro Aufruf."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, Optional, TextIO


class ToolCallLogger:
    """Schreibt für jeden Tool-Call einen strukturierten JSON-Log-Eintrag."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream or sys.stderr

    def log(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        *,
        output: Optional[str] = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
        attempt: int = 0,
        retryable: Optional[bool] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "tool_name": tool_name,
            "input": tool_input,
            "duration_ms": round(duration_ms, 2),
            "attempt": attempt,
        }

        if error is not None:
            record["status"] = "error"
            record["error"] = error
            if retryable is not None:
                record["retryable"] = retryable
        else:
            record["status"] = "ok"
            record["output"] = output

        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
