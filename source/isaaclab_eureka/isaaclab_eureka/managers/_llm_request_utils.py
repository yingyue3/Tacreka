import os
import time
from typing import Any

import openai


DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_LLM_REQUEST_RETRIES = 3
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 5.0


def _read_positive_float_env(var_name: str, default: float) -> float:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        print(f"[WARNING]: Invalid {var_name}={raw_value!r}; using default {default}.")
        return default
    if value <= 0.0:
        print(f"[WARNING]: {var_name} must be > 0; using default {default}.")
        return default
    return value


def _read_nonnegative_float_env(var_name: str, default: float) -> float:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        print(f"[WARNING]: Invalid {var_name}={raw_value!r}; using default {default}.")
        return default
    if value < 0.0:
        print(f"[WARNING]: {var_name} must be >= 0; using default {default}.")
        return default
    return value


def _read_nonnegative_int_env(var_name: str, default: int) -> int:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        print(f"[WARNING]: Invalid {var_name}={raw_value!r}; using default {default}.")
        return default
    if value < 0:
        print(f"[WARNING]: {var_name} must be >= 0; using default {default}.")
        return default
    return value


def get_llm_request_settings() -> tuple[float, int, float]:
    """Resolve shared LLM timeout/retry settings from environment variables."""
    timeout_seconds = _read_positive_float_env(
        "ISAACLAB_EUREKA_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    )
    max_request_retries = _read_nonnegative_int_env(
        "ISAACLAB_EUREKA_LLM_MAX_RETRIES", DEFAULT_LLM_REQUEST_RETRIES
    )
    retry_backoff_seconds = _read_nonnegative_float_env(
        "ISAACLAB_EUREKA_LLM_RETRY_BACKOFF_SECONDS", DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    )
    return timeout_seconds, max_request_retries, retry_backoff_seconds


def build_openai_client(timeout_seconds: float):
    """Create an OpenAI/Azure OpenAI client with bounded request duration."""
    client_kwargs = {
        "timeout": timeout_seconds,
        # We handle retries explicitly so logs and timing stay predictable.
        "max_retries": 0,
    }
    if "AZURE_OPENAI_API_KEY" in os.environ:
        return openai.AzureOpenAI(api_version="2024-02-01", **client_kwargs)
    if "OPENAI_API_KEY" in os.environ:
        return openai.OpenAI(**client_kwargs)
    raise RuntimeError("No Openai API key found in environment variables")


def _should_retry(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in {408, 409, 429} or (status_code is not None and status_code >= 500)
    return False


def create_chat_completion(
    client,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    n: int,
    timeout_seconds: float,
    max_request_retries: int,
    retry_backoff_seconds: float,
    request_name: str,
    response_format: dict[str, Any] | None = None,
):
    """Run a chat completion with bounded retries for transient failures."""
    max_attempts = max(1, max_request_retries + 1)
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "n": n,
        "timeout": timeout_seconds,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format

    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            is_retryable = _should_retry(exc)
            if not is_retryable or attempt >= max_attempts:
                raise
            sleep_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[WARNING]: {request_name} failed on attempt {attempt}/{max_attempts} "
                f"with {exc.__class__.__name__}: {exc}. Retrying in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)
