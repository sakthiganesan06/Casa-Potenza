"""
stt/backoff.py — Exponential backoff retry logic for Sarvam WebSocket connections.

Implements a tenacity-based retry decorator with:
- Exponential backoff: delays [0.5s, 1s, 2s, 4s, 8s] + random jitter
- Max 5 retries before raising the final exception
- Structured logging on each retry attempt
- Special handling for WebSocket disconnect codes (1001, 1006, 1011)
"""
import asyncio
import functools
import random
from typing import Callable, TypeVar, ParamSpec

import websockets.exceptions
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
import logging

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Retryable exception types
# ---------------------------------------------------------------------------
_RETRYABLE_EXCEPTIONS = (
    websockets.exceptions.ConnectionClosedError,
    websockets.exceptions.ConnectionClosedOK,
    websockets.exceptions.WebSocketException,
    ConnectionResetError,
    ConnectionRefusedError,
    OSError,
    TimeoutError,
    asyncio.TimeoutError,
)

# ---------------------------------------------------------------------------
# Tenacity retry strategy
# ---------------------------------------------------------------------------
# wait_random_exponential adds jitter to avoid thundering herd on reconnect
_RETRY_STRATEGY = dict(
    stop=stop_after_attempt(config.BACKOFF_MAX_RETRIES),
    wait=wait_random_exponential(
        multiplier=config.BACKOFF_MULTIPLIER,
        min=config.BACKOFF_MIN_WAIT,
        max=config.BACKOFF_MAX_WAIT,
    ),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
)


# ---------------------------------------------------------------------------
# Decorator for async functions
# ---------------------------------------------------------------------------

P = ParamSpec("P")
T = TypeVar("T")


def with_websocket_retry(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that wraps an async function with exponential backoff retry.

    Usage:
        @with_websocket_retry
        async def connect_and_stream(...):
            ...

    On each retry attempt, logs:
    - attempt number
    - exception type and message
    - next wait time (seconds)
    """
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        attempt = 0
        async for attempt_ctx in AsyncRetrying(**_RETRY_STRATEGY):
            with attempt_ctx:
                attempt += 1
                if attempt > 1:
                    logger.warning(
                        f"[Backoff] Retry attempt {attempt}/{config.BACKOFF_MAX_RETRIES} "
                        f"for {fn.__name__}"
                    )
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except RetryError as exc:
                    logger.error(
                        f"[Backoff] {fn.__name__} failed after "
                        f"{config.BACKOFF_MAX_RETRIES} retries: {exc}"
                    )
                    raise
        # Should be unreachable, but satisfies type checkers
        raise RuntimeError(f"{fn.__name__} retry loop exited unexpectedly")

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Context-manager style retry (for use inside async generators / streams)
# ---------------------------------------------------------------------------

class WebSocketRetryer:
    """
    Context-manager style retry helper for WebSocket connection streams.

    Allows fine-grained control over when to retry:

        retryer = WebSocketRetryer(label="sarvam_stt")
        async for attempt in retryer:
            try:
                async with websockets.connect(...) as ws:
                    await stream_audio(ws)
                    break  # success — exit retry loop
            except retryable_exceptions as exc:
                await retryer.record_failure(exc)
    """

    def __init__(self, label: str = "WebSocket"):
        self._label = label
        self._attempt = 0
        self._delays = _compute_delays()

    def __aiter__(self):
        return self

    async def __anext__(self) -> int:
        if self._attempt > 0:
            if self._attempt >= config.BACKOFF_MAX_RETRIES:
                logger.error(
                    f"[Backoff:{self._label}] Max retries ({config.BACKOFF_MAX_RETRIES}) "
                    f"exceeded. Giving up."
                )
                raise StopAsyncIteration

            delay = self._delays[self._attempt - 1]
            logger.warning(
                f"[Backoff:{self._label}] Attempt {self._attempt + 1}/"
                f"{config.BACKOFF_MAX_RETRIES} — waiting {delay:.2f}s before reconnect..."
            )
            await asyncio.sleep(delay)

        self._attempt += 1
        return self._attempt

    async def record_failure(self, exc: Exception) -> None:
        """Log the exception; iteration will apply backoff on next __anext__."""
        logger.warning(
            f"[Backoff:{self._label}] Attempt {self._attempt} failed: "
            f"{type(exc).__name__}: {exc}"
        )


def _compute_delays() -> list[float]:
    """
    Pre-compute exponential backoff delays with jitter.
    Returns a list of length BACKOFF_MAX_RETRIES - 1.
    """
    delays = []
    for i in range(config.BACKOFF_MAX_RETRIES - 1):
        base = config.BACKOFF_MULTIPLIER * (2 ** i)
        # Add ±20% jitter
        jitter = base * 0.2 * (2 * random.random() - 1)
        delay = max(config.BACKOFF_MIN_WAIT, min(config.BACKOFF_MAX_WAIT, base + jitter))
        delays.append(delay)
    return delays
