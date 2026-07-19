import time

import pytest

from harness.sandbox import Sandbox, SandboxConfig, SandboxViolation, ToolTimeoutError


def test_run_within_timeout_returns_result():
    sandbox = Sandbox(SandboxConfig(default_timeout_seconds=1.0))
    try:
        result = sandbox.run(lambda x: x * 2, 21)
        assert result == 42
    finally:
        sandbox.shutdown()


def test_run_exceeding_timeout_raises_tool_timeout_error():
    sandbox = Sandbox(SandboxConfig(default_timeout_seconds=0.1))

    def slow():
        time.sleep(1.0)
        return "done"

    try:
        with pytest.raises(ToolTimeoutError):
            sandbox.run(slow)
    finally:
        sandbox.shutdown()


def test_per_call_timeout_override_is_respected():
    sandbox = Sandbox(SandboxConfig(default_timeout_seconds=5.0))

    def slow():
        time.sleep(1.0)
        return "done"

    try:
        with pytest.raises(ToolTimeoutError):
            sandbox.run(slow, timeout_seconds=0.1)
    finally:
        sandbox.shutdown()


def test_domain_allowlist_blocks_disallowed_domain():
    sandbox = Sandbox(SandboxConfig(allowed_domains=frozenset({"example.com"})))
    try:
        with pytest.raises(SandboxViolation):
            sandbox.check_domain_allowed("https://evil.example.org/data")
    finally:
        sandbox.shutdown()


def test_domain_allowlist_allows_subdomains():
    sandbox = Sandbox(SandboxConfig(allowed_domains=frozenset({"example.com"})))
    try:
        sandbox.check_domain_allowed("https://docs.example.com/page")  # darf nicht werfen
    finally:
        sandbox.shutdown()


def test_no_allowlist_configured_allows_everything():
    sandbox = Sandbox(SandboxConfig(allowed_domains=None))
    try:
        sandbox.check_domain_allowed("https://anything.example")  # darf nicht werfen
    finally:
        sandbox.shutdown()


def test_enforce_max_size_truncates_long_content():
    sandbox = Sandbox(SandboxConfig(max_response_bytes=10))
    try:
        result = sandbox.enforce_max_size("0123456789ABCDEF")
        assert result.startswith("0123456789")
        assert "gekürzt" in result
    finally:
        sandbox.shutdown()


def test_enforce_max_size_leaves_short_content_untouched():
    sandbox = Sandbox(SandboxConfig(max_response_bytes=1000))
    try:
        text = "kurzer Text"
        assert sandbox.enforce_max_size(text) == text
    finally:
        sandbox.shutdown()
