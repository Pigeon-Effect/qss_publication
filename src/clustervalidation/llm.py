"""Thin wrapper around the DeepSeek chat-completions API.

The API is OpenAI-compatible, so the official ``openai`` client is used with a
different base URL. This module exists to keep three concerns out of the
protocol code: credential handling, retry behaviour, and cost accounting.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from clustervalidation.config import DEFAULT_BASE_URL, PROJECT_ROOT, ModelSpec

# The .env file lives at the repository root. Its location is derived from the
# package rather than looked up from the caller: bare load_dotenv() searches
# upward from the calling frame, which makes the result depend on the working
# directory and raises AssertionError when there is no parent frame.
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")

API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

# How long to wait before trying again when the API cannot be reached at all.
DEFAULT_OUTAGE_WAIT_SECONDS = 600.0

# The API is unreachable: the network is down, the request timed out, or the
# service is not serving. None of these is the run's fault and all of them
# resolve on their own, so they are waited out indefinitely rather than
# consuming the retry budget. A long run on an unreliable connection should
# pause, not lose trials.
_UNREACHABLE_ERRORS = (
    APIConnectionError,   # includes DNS failure and refused connections
    APITimeoutError,      # subclass of APIConnectionError, listed for clarity
    RateLimitError,       # 429: serving, but not to us, and only for now
    InternalServerError,  # 5xx
)

# These will never succeed on retry: the credentials, the model name or the
# request itself is wrong. Retrying endlessly would turn a typo into a silent
# hang, so they are raised immediately.
_FATAL_ERRORS = (
    AuthenticationError,     # 401
    PermissionDeniedError,   # 403
    BadRequestError,         # 400
    NotFoundError,           # 404, e.g. an unknown model name
)

_MISSING_KEY_MESSAGE = f"""\
No API key found. Set {API_KEY_ENV_VAR} before running an experiment, either
in a local .env file (copy .env.example) or in the environment directly:

  .env file    {API_KEY_ENV_VAR}=sk-...
  PowerShell   $env:{API_KEY_ENV_VAR} = 'sk-...'
  bash/zsh     export {API_KEY_ENV_VAR}='sk-...'

Keys are issued at https://platform.deepseek.com. .env is gitignored; never
commit a key to the repository - see .env.example.\
"""


class MissingAPIKeyError(RuntimeError):
    """Raised when no API key is available in the environment."""


@dataclass(frozen=True)
class Completion:
    """One model response, with the metadata the reports need."""

    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int

    @property
    def truncated(self) -> bool:
        """True when the response hit the token ceiling and was cut off."""
        return self.finish_reason == "length"


def resolve_api_key(explicit: str | None = None) -> str:
    """Return the API key, preferring an explicit value over the environment.

    A local ``.env`` file is loaded first, if one exists. Variables already
    present in the real environment win, so an exported key still overrides
    the file.
    """
    load_dotenv(DOTENV_PATH)
    key = explicit or os.environ.get(API_KEY_ENV_VAR)
    if not key:
        raise MissingAPIKeyError(_MISSING_KEY_MESSAGE)
    return key


class ChatClient:
    """Issues chat completions for a fixed model specification."""

    def __init__(
        self,
        model: ModelSpec,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        outage_wait_seconds: float = DEFAULT_OUTAGE_WAIT_SECONDS,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.outage_wait_seconds = outage_wait_seconds
        # Counted across the client's lifetime and reported at the end of a
        # run, so a result carries evidence of how disrupted its network was.
        self.outage_waits = 0
        self._client = OpenAI(api_key=resolve_api_key(api_key), base_url=base_url)

    def _report_outage(self, error: Exception) -> None:
        """Announce a wait on stderr so a paused run is not mistaken for a hung one."""
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        minutes = self.outage_wait_seconds / 60.0
        print(
            f"[{stamp}] API unreachable ({type(error).__name__}). "
            f"Waiting {minutes:.0f} min, then retrying "
            f"(outage wait #{self.outage_waits}). Ctrl-C to abort.",
            file=sys.stderr,
            flush=True,
        )

    def complete(
        self,
        prompt: str,
        system_message: str | None = None,
        model: ModelSpec | None = None,
    ) -> Completion:
        """Send one prompt and return the parsed response.

        Transient API failures are retried with exponential backoff. The final
        failure is raised to the caller, which records it as a failed trial
        rather than aborting the run.

        ``model`` overrides the client's own specification for this call alone,
        which the forced-choice follow-up uses to ask a cheap, non-reasoning
        model for a verdict without opening a second connection.
        """
        spec = model or self.model
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        request: dict = {
            "model": spec.name,
            "messages": messages,
            "max_tokens": spec.max_tokens,
            "timeout": spec.timeout,
        }
        if spec.temperature is not None:
            request["temperature"] = spec.temperature
        if spec.thinking:
            request["extra_body"] = {"thinking": {"type": "enabled"}}

        last_error: Exception | None = None
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self._client.chat.completions.create(**request)
            except _FATAL_ERRORS:
                # Wrong key, wrong model, malformed request: no amount of
                # waiting fixes these, and pretending otherwise would hang.
                raise
            except _UNREACHABLE_ERRORS as error:
                # Wait the outage out. This does not consume `attempt`, so a
                # network that is down for hours costs time but not trials.
                self.outage_waits += 1
                self._report_outage(error)
                time.sleep(self.outage_wait_seconds)
                continue
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                attempt += 1
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
                continue

            choice = response.choices[0]
            usage = response.usage
            return Completion(
                content=(choice.message.content or "").strip(),
                reasoning=(
                    getattr(choice.message, "reasoning_content", "") or ""
                ).strip(),
                finish_reason=choice.finish_reason or "",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=spec.cost(usage.prompt_tokens, usage.completion_tokens),
                attempts=attempt,
            )

        raise RuntimeError(
            f"request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
