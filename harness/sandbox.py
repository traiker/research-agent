"""Kontrolliert die Ausführung einzelner Tool-Calls: harte Timeouts pro Call,
eine Domain-Allowlist (für Tools wie web_fetch) und eine Obergrenze für die
Antwortgröße."""

from __future__ import annotations

import dataclasses
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, FrozenSet, Optional


class SandboxViolation(Exception):
    """Basisklasse: ein Tool-Call hat eine Sandbox-Policy verletzt."""


class ToolTimeoutError(SandboxViolation):
    """Der Tool-Call hat sein Zeitlimit überschritten."""


@dataclasses.dataclass
class SandboxConfig:
    default_timeout_seconds: float = 15.0
    # None = alle Domains erlaubt (keine Einschränkung konfiguriert)
    allowed_domains: Optional[FrozenSet[str]] = None
    max_response_bytes: int = 200_000


class Sandbox:
    """Erzwingt Ressourcenlimits rund um die Tool-Ausführung."""

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="tool-call"
        )

    def run(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Führt `func` in einem Worker-Thread aus und erzwingt ein Timeout.

        Hinweis: `future.cancel()` stoppt keinen bereits laufenden Thread
        (Python-Threads sind nicht sicher unterbrechbar) - der Aufrufer bekommt
        aber sofort einen ToolTimeoutError und der Agent-Loop kann weitermachen,
        statt an einem hängenden Tool-Call zu blockieren.
        """
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.config.default_timeout_seconds
        )
        future = self._executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise ToolTimeoutError(
                f"Tool-Call hat das Timeout von {timeout}s überschritten"
            ) from None

    def check_domain_allowed(self, url: str) -> None:
        """Wirft SandboxViolation, wenn die Host-Domain nicht auf der Allowlist steht."""
        if self.config.allowed_domains is None:
            return

        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if not host:
            raise SandboxViolation(f"URL ohne gültigen Host: {url!r}")

        allowed = any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.config.allowed_domains
        )
        if not allowed:
            raise SandboxViolation(
                f"Domain nicht auf der Sandbox-Allowlist erlaubt: {host!r}"
            )

    def enforce_max_size(self, content: str) -> str:
        """Kürzt `content` auf `max_response_bytes` (UTF-8), falls nötig."""
        limit = self.config.max_response_bytes
        encoded = content.encode("utf-8", errors="ignore")
        if len(encoded) <= limit:
            return content

        truncated = encoded[:limit].decode("utf-8", errors="ignore")
        return truncated + f"\n\n[... gekürzt, Antwort überschritt {limit} Bytes ...]"

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
