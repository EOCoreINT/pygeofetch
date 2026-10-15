"""
Retry logic for PyGeoFetch using tenacity.

Provides decorators and utilities for retrying failed HTTP requests and
provider API calls with configurable backoff strategies.

Example::

    from pygeofetch.utils.retry_handler import retry_on_failure, RetryConfig

    config = RetryConfig(attempts=5, strategy="exponential_jitter")

    @retry_on_failure(config)
    def fetch_data():
        ...
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

from typing_extensions import Self

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger("pygeofetch.retry")


class RetryConfig:
    """
    Configuration for retry behaviour.

    Attributes:
        attempts: Maximum total attempts (including first try).
        strategy: Backoff strategy name.
        delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Whether to add random jitter.
        reraise_on: Exception types that should NOT be retried.
    """

    def __init__(
        self,
        attempts: int = 3,
        strategy: str = "exponential_jitter",
        delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        reraise_on: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.attempts = max(1, attempts)
        self.strategy = strategy
        self.delay = delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.reraise_on = reraise_on or ()

    def get_delay(self, attempt: int) -> float:
        """
        Compute the delay for the given attempt number (0-indexed).

        Args:
            attempt: Current attempt index (0 = first retry).

        Returns:
            Seconds to wait before next attempt.
        """
        if self.strategy == "fixed":
            delay = self.delay
        elif self.strategy == "linear":
            delay = self.delay * (attempt + 1)
        elif self.strategy in ("exponential", "exponential_jitter"):
            delay = self.delay * (2**attempt)
        else:
            delay = self.delay

        delay = min(delay, self.max_delay)

        if self.jitter or self.strategy == "exponential_jitter":
            delay *= 0.5 + random.random() * 0.5

        return delay


def retry_on_failure(
    config: RetryConfig | None = None,
    attempts: int = 3,
    strategy: str = "exponential_jitter",
) -> Callable[[F], F]:
    """
    Decorator that retries a function on exception.

    Args:
        config: RetryConfig instance (overrides other args if provided).
        attempts: Number of attempts if config not provided.
        strategy: Backoff strategy if config not provided.

    Returns:
        Decorator function.

    Example::

        @retry_on_failure(attempts=5, strategy="exponential")
        def unstable_api_call():
            response = requests.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()
    """
    cfg = config or RetryConfig(attempts=attempts, strategy=strategy)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(cfg.attempts):
                try:
                    return func(*args, **kwargs)
                except cfg.reraise_on:
                    raise
                except Exception as exc:
                    last_exception = exc
                    if attempt < cfg.attempts - 1:
                        delay = cfg.get_delay(attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{cfg.attempts}): "
                            f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {cfg.attempts} attempts: "
                            f"{type(exc).__name__}: {exc}"
                        )
            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for failing providers.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Provider is failing; requests are blocked immediately.
        HALF_OPEN: Testing recovery; one request allowed through.

    Example::

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        with breaker:
            result = call_provider_api()
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "unknown",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None

    @property
    def state(self) -> str:
        """Return current circuit breaker state."""
        if self._state == self.OPEN:
            if self._last_failure_time and (
                time.monotonic() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = self.HALF_OPEN
        return self._state

    def __enter__(self) -> Self:
        state = self.state
        if state == self.OPEN:
            msg = (
                f"Circuit breaker for '{self.name}' is OPEN. "
                f"Provider appears to be down. Try again later."
            )
            raise CircuitBreakerOpenError(msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._on_success()
        elif exc_type is not CircuitBreakerOpenError:
            self._on_failure()
        return None

    def _on_success(self) -> None:
        self._failure_count = 0
        self._state = self.CLOSED

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            if self._state != self.OPEN:
                logger.warning(
                    f"Circuit breaker for '{self.name}' is now OPEN after "
                    f"{self._failure_count} failures."
                )
            self._state = self.OPEN

    @property
    def is_open(self) -> bool:
        """Return True if circuit breaker is open (blocking requests)."""
        return self.state == self.OPEN


class CircuitBreakerOpenError(Exception):
    """Raised when a request is blocked by an open circuit breaker."""
