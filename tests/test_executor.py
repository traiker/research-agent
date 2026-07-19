import time
from unittest.mock import MagicMock

from harness.executor import ToolExecutor, TransientToolError
from harness.logging import ToolCallLogger
from harness.registry import Tool, ToolRegistry
from harness.sandbox import Sandbox, SandboxConfig, SandboxViolation


def make_executor(max_retries: int = 2, default_timeout_seconds: float = 1.0):
    sandbox = Sandbox(SandboxConfig(default_timeout_seconds=default_timeout_seconds))
    registry = ToolRegistry()
    logger = MagicMock(spec=ToolCallLogger)
    executor = ToolExecutor(
        registry, sandbox, logger, max_retries=max_retries, base_backoff_seconds=0.01
    )
    return registry, executor, sandbox, logger


def test_successful_call_returns_result_and_logs_ok():
    registry, executor, sandbox, logger = make_executor()
    try:
        registry.register(
            Tool(name="echo", description="", input_schema={}, handler=lambda inp: inp["value"])
        )

        result = executor.execute("tool_1", "echo", {"value": "hi"})

        assert result.content == "hi"
        assert result.is_error is False
        logger.log.assert_called_once()
    finally:
        sandbox.shutdown()


def test_unknown_tool_returns_structured_error_without_raising():
    registry, executor, sandbox, logger = make_executor()
    try:
        result = executor.execute("tool_1", "does_not_exist", {})

        assert result.is_error is True
        assert "unbekanntes Tool" in result.content
    finally:
        sandbox.shutdown()


def test_handler_exception_is_caught_and_returned_as_structured_error():
    registry, executor, sandbox, logger = make_executor()
    try:

        def boom(_inp):
            raise ValueError("kaboom")

        registry.register(Tool(name="boom", description="", input_schema={}, handler=boom))

        result = executor.execute("tool_1", "boom", {})

        assert result.is_error is True
        assert "kaboom" in result.content
    finally:
        sandbox.shutdown()


def test_sandbox_violation_is_not_retried():
    registry, executor, sandbox, logger = make_executor(max_retries=2)
    try:
        call_count = {"n": 0}

        def violating(_inp):
            call_count["n"] += 1
            raise SandboxViolation("Domain nicht erlaubt")

        registry.register(Tool(name="blocked", description="", input_schema={}, handler=violating))

        result = executor.execute("tool_1", "blocked", {})

        assert result.is_error is True
        assert call_count["n"] == 1  # kein Retry bei Policy-Verletzungen
    finally:
        sandbox.shutdown()


def test_transient_error_is_retried_and_eventually_succeeds():
    registry, executor, sandbox, logger = make_executor(max_retries=2)
    try:
        call_count = {"n": 0}

        def flaky(_inp):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise TransientToolError("kurzzeitiger Ausfall")
            return "recovered"

        registry.register(Tool(name="flaky", description="", input_schema={}, handler=flaky))

        result = executor.execute("tool_1", "flaky", {})

        assert result.is_error is False
        assert result.content == "recovered"
        assert call_count["n"] == 2
    finally:
        sandbox.shutdown()


def test_transient_error_exhausts_retries_and_returns_structured_error():
    registry, executor, sandbox, logger = make_executor(max_retries=2)
    try:
        call_count = {"n": 0}

        def always_flaky(_inp):
            call_count["n"] += 1
            raise TransientToolError("weiterhin kaputt")

        registry.register(
            Tool(name="always_flaky", description="", input_schema={}, handler=always_flaky)
        )

        result = executor.execute("tool_1", "always_flaky", {})

        assert result.is_error is True
        assert call_count["n"] == 3  # initialer Versuch + 2 Retries
    finally:
        sandbox.shutdown()


def test_timeout_is_treated_as_transient_and_retried():
    registry, executor, sandbox, logger = make_executor(max_retries=1, default_timeout_seconds=0.1)
    try:
        call_count = {"n": 0}

        def slow_then_fast(_inp):
            call_count["n"] += 1
            if call_count["n"] == 1:
                time.sleep(0.5)
            return "fast now"

        registry.register(
            Tool(
                name="slow",
                description="",
                input_schema={},
                handler=slow_then_fast,
                timeout_seconds=0.1,
            )
        )

        result = executor.execute("tool_1", "slow", {})

        assert result.is_error is False
        assert result.content == "fast now"
        assert call_count["n"] == 2
    finally:
        sandbox.shutdown()
