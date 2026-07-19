"""CLI-Einstiegspunkt für den Research-Agenten.

Beispiel:
    cp .env.example .env   # ANTHROPIC_API_KEY dort eintragen
    python main.py "Was ist die Hauptstadt von Frankreich?"
"""

from __future__ import annotations

import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv

from agent.loop import DEFAULT_MODEL, MAX_ITERATIONS, AgentLoop
from harness.executor import ToolExecutor
from harness.logging import ToolCallLogger
from harness.registry import Tool, ToolRegistry
from harness.sandbox import Sandbox, SandboxConfig
from tools import web_fetch, web_search

load_dotenv()  # liest ANTHROPIC_API_KEY (und weitere Variablen) aus einer lokalen .env

DEFAULT_ALLOWED_DOMAINS = frozenset(
    {
        "en.wikipedia.org",
        "de.wikipedia.org",
        "docs.python.org",
        "duckduckgo.com",
    }
)

SYSTEM_PROMPT = (
    "Du bist ein Research-Agent. Nutze die verfügbaren Tools (web_search, "
    "web_fetch), um Fragen faktenbasiert zu beantworten. Nenne kurz die "
    "Quellen, die du verwendet hast."
)


def build_registry(sandbox: Sandbox) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_search",
            description=web_search.DESCRIPTION,
            input_schema=web_search.INPUT_SCHEMA,
            handler=web_search.handler,
            timeout_seconds=15.0,
        )
    )
    registry.register(
        Tool(
            name="web_fetch",
            description=web_fetch.DESCRIPTION,
            input_schema=web_fetch.INPUT_SCHEMA,
            handler=web_fetch.make_handler(sandbox),
            timeout_seconds=15.0,
        )
    )
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-Agent mit Tool-Harness")
    parser.add_argument("query", help="Die Recherche-Frage an den Agenten")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Anthropic Modell-ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="max_tokens pro Antwort (default: 4096)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Sicherheitsgrenze für Loop-Iterationen (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=None,
        metavar="DOMAIN",
        help="Zusätzliche für web_fetch erlaubte Domain (mehrfach nutzbar)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY ist nicht gesetzt.", file=sys.stderr)
        sys.exit(1)

    allowed_domains = set(DEFAULT_ALLOWED_DOMAINS)
    if args.allow_domain:
        allowed_domains.update(args.allow_domain)

    sandbox = Sandbox(SandboxConfig(allowed_domains=frozenset(allowed_domains)))
    registry = build_registry(sandbox)
    logger = ToolCallLogger()
    executor = ToolExecutor(registry, sandbox, logger, max_retries=2)

    client = anthropic.Anthropic()
    loop = AgentLoop(
        client=client,
        executor=executor,
        tools=registry.to_api_tools(),
        model=args.model,
        system=SYSTEM_PROMPT,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
    )

    try:
        answer = loop.run(args.query)
    finally:
        sandbox.shutdown()

    print(answer)


if __name__ == "__main__":
    main()
