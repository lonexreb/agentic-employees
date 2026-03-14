"""Shared test fixtures and markers."""

import asyncio

import pytest


def _nats_available() -> bool:
    """Check if NATS is reachable at localhost:4222 with a 1s timeout."""

    async def _check():
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", 4222),
                timeout=1.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (OSError, asyncio.TimeoutError):
            return False

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_check())
    finally:
        loop.close()


NATS_AVAILABLE = _nats_available()

requires_nats = pytest.mark.skipif(
    not NATS_AVAILABLE,
    reason="NATS server not running at localhost:4222",
)
