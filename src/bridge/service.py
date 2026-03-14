"""BridgeService — connects HTTP API to NATS event bus."""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from src.config import Config
from src.events.bus import EventBus
from src.events.topics import result_topic
from src.events.types import ResultEvent

from .http_api import create_app

logger = logging.getLogger(__name__)


class BridgeService:
    """HTTP ←→ NATS bridge for OpenClaw agents.

    OpenClaw agents call HTTP endpoints (via exec/curl).  The bridge
    translates those into NATS pub/sub events so PRM Evaluator and
    Training Loop receive identical Pydantic JSON on the same topics.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.bus = EventBus()
        self._results: dict[str, ResultEvent] = {}
        self._waiters: dict[str, asyncio.Event] = {}

    async def start(self) -> None:
        await self.bus.connect(self.cfg.nats_url)

        # Subscribe to all results so we can relay back to polling managers
        await self.bus.subscribe(
            result_topic("*"), ResultEvent, self._on_result,
        )

        app = create_app(self.bus, self._results, self._waiters)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.cfg.bridge_port)
        await site.start()
        logger.info("Bridge HTTP API listening on :%s", self.cfg.bridge_port)

        # Block forever
        await asyncio.Event().wait()

    async def _on_result(self, result: ResultEvent) -> None:
        self._results[result.task_id] = result
        waiter = self._waiters.get(result.task_id)
        if waiter:
            waiter.set()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
    )
    bridge = BridgeService()
    await bridge.start()


if __name__ == "__main__":
    asyncio.run(main())
