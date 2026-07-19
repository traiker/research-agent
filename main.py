"""CLI-Einstiegspunkt für den Research-Agenten.

Beispiel:
    cp .env.example .env   # ANTHROPIC_API_KEY und/oder OPENAI_API_KEY eintragen
    python main.py "Was ist die Hauptstadt von Frankreich?"
    python main.py --provider openai "Was ist die Hauptstadt von Frankreich?"
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from agent.loop import MAX_ITERATIONS, AgentLoop
from agent.providers.base import AgentProvider
from agent.providers.claude_provider import ClaudeProvider
from agent.providers.claude_provider import DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL
from agent.providers.openai_provider import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from agent.providers.openai_provider import OpenAIProvider
from harness.executor import ToolExecutor
from harness.logging import ToolCallLogger
from harness.registry import Tool, ToolRegistry
from harness.sandbox import Sandbox, SandboxConfig
from tools import web_fetch, web_search

load_dotenv()  # liest ANTHROPIC_API_KEY / OPENAI_API_KEY (und weitere Variablen) aus einer lokalen .env

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

PROVIDER_DEFAULT_MODELS = {
    "claude": CLAUDE_DEFAULT_MODEL,
    "openai": OPENAI_DEFAULT_MODEL,
}

REQUIRED_ENV_VAR = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


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


def build_provider(
    provider_name: str, model: str, tools_schema: list, max_tokens: int
) -> AgentProvider:
    if provider_name == "claude":
        import anthropic

        return ClaudeProvider(
            client=anthropic.Anthropic(),
            model=model,
            tools_schema=tools_schema,
            system=SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
    if provider_name == "openai":
        from openai import OpenAI

        return OpenAIProvider(
            client=OpenAI(),
            model=model,
            tools_schema=tools_schema,
            system=SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unbekannter Provider: {provider_name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-Agent mit Tool-Harness")
    parser.add_argument("query", help="Die Recherche-Frage an den Agenten")
    parser.add_argument(
        "--provider",
        choices=["claude", "openai"],
        default="claude",
        help="Welche LLM-API genutzt werden soll (default: claude)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modell-ID (default: providerabhängig, siehe PROVIDER_DEFAULT_MODELS)",
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

    required_var = REQUIRED_ENV_VAR[args.provider]
    if not os.environ.get(required_var):
        print(
            f"{required_var} ist nicht gesetzt (benötigt für --provider {args.provider}).",
            file=sys.stderr,
        )
        sys.exit(1)

    model = args.model or PROVIDER_DEFAULT_MODELS[args.provider]

    allowed_domains = set(DEFAULT_ALLOWED_DOMAINS)
    if args.allow_domain:
        allowed_domains.update(args.allow_domain)

    sandbox = Sandbox(SandboxConfig(allowed_domains=frozenset(allowed_domains)))
    registry = build_registry(sandbox)
    logger = ToolCallLogger()
    executor = ToolExecutor(registry, sandbox, logger, max_retries=2)

    provider = build_provider(args.provider, model, registry.to_api_tools(), args.max_tokens)
    loop = AgentLoop(provider=provider, executor=executor, max_iterations=args.max_iterations)

    try:
        answer = loop.run(args.query)
    finally:
        sandbox.shutdown()

    print(answer)


if __name__ == "__main__":
    main()
