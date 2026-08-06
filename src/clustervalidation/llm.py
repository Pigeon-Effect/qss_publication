"""Thin wrapper around the DeepSeek chat-completions API.

The API is OpenAI-compatible, so the official ``openai`` client is used with a
different base URL. This module exists to keep three concerns out of the
protocol code: credential handling, retry behaviour, and cost accounting.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from clustervalidation.config import DEFAULT_BASE_URL, ModelSpec

API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

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
    load_dotenv()
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
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._client = OpenAI(api_key=resolve_api_key(api_key), base_url=base_url)

    def complete(self, prompt: str, system_message: str | None = None) -> Completion:
        """Send one prompt and return the parsed response.

        Transient API failures are retried with exponential backoff. The final
        failure is raised to the caller, which records it as a failed trial
        rather than aborting the run.
        """
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        request: dict = {
            "model": self.model.name,
            "messages": messages,
            "max_tokens": self.model.max_tokens,
            "timeout": self.model.timeout,
        }
        if self.model.temperature is not None:
            request["temperature"] = self.model.temperature
        if self.model.thinking:
            request["extra_body"] = {"thinking": {"type": "enabled"}}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
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
                cost_usd=self.model.cost(usage.prompt_tokens, usage.completion_tokens),
                attempts=attempt,
            )

        raise RuntimeError(
            f"request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
