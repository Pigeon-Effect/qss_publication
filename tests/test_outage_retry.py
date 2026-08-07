"""The client must sit out network outages rather than lose trials.

A thousand-trial run takes hours, so a dropped connection has to pause it, not
abort it. These tests pin the two halves of that contract: transient
unreachability is waited out indefinitely, and errors that will never succeed
are raised at once instead of hanging the run forever.

No network access and no API key are required.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from clustervalidation.config import MODELS
from clustervalidation.llm import ChatClient

_REQUEST = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


class _FailThenSucceed:
    """Raise ``error`` for the first ``failures`` calls, then return a completion."""

    def __init__(self, error: Exception, failures: int) -> None:
        self.error = error
        self.failures = failures
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error

        class _Message:
            content = "Final verdict: 2"
            reasoning_content = ""

        class _Choice:
            message = _Message()
            finish_reason = "stop"

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Response:
            choices = [_Choice()]
            usage = _Usage()

        return _Response()


def _client() -> ChatClient:
    # A negligible wait keeps the tests fast; the retry logic is identical.
    return ChatClient(
        MODELS["deepseek-chat"], api_key="sk-test", outage_wait_seconds=0.0
    )


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=_REQUEST),
        RateLimitError("429", response=_response(429), body=None),
        InternalServerError("500", response=_response(500), body=None),
    ],
    ids=["connection", "rate_limit", "server_error"],
)
def test_unreachable_api_is_waited_out(error: Exception) -> None:
    """Transient unreachability costs time, never a trial."""
    client = _client()
    stub = _FailThenSucceed(error, failures=4)
    with patch.object(client._client.chat.completions, "create", stub):
        completion = client.complete("prompt")
    assert completion.content == "Final verdict: 2"
    assert client.outage_waits == 4


def test_outage_retry_does_not_consume_the_retry_budget() -> None:
    """Far more outages than max_retries must still recover.

    This is the property the long runs depend on: an hour-long outage is 6
    waits at the default interval, well past the 3-attempt budget that governs
    ordinary errors.
    """
    client = _client()
    stub = _FailThenSucceed(APIConnectionError(request=_REQUEST), failures=50)
    with patch.object(client._client.chat.completions, "create", stub):
        completion = client.complete("prompt")
    assert completion.content == "Final verdict: 2"
    assert client.outage_waits == 50
    assert client.max_retries == 3


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("401", response=_response(401), body=None),
        BadRequestError("400", response=_response(400), body=None),
        NotFoundError("404", response=_response(404), body=None),
    ],
    ids=["auth", "bad_request", "not_found"],
)
def test_fatal_errors_raise_immediately(error: Exception) -> None:
    """A wrong key or model name must fail loudly, not retry until the heat death."""
    client = _client()
    stub = _FailThenSucceed(error, failures=999)
    with patch.object(client._client.chat.completions, "create", stub):
        with pytest.raises(type(error)):
            client.complete("prompt")
    assert stub.calls == 1
    assert client.outage_waits == 0
