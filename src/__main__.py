"""Demo entry point: python -m src"""

import asyncio
import logging

from src.config import Config
from src.events.bus import EventBus
from src.manager.manager import Manager
from src.workers.echo_worker import EchoWorker


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config()

    bus = EventBus()
    await bus.connect(cfg.nats_url)

    manager = Manager(cfg.manager_id, bus)
    worker = EchoWorker(cfg.worker_id, bus)

    await manager.start()
    await worker.start()

    print("\n--- Assigning task ---")
    task = await manager.assign_task("coding", "Write a fibonacci function in Python")
    print(f"Task ID: {task.task_id}")

    result = await manager.wait_for_result(task.task_id, timeout=cfg.task_timeout_seconds)
    print(f"\n--- Result ---")
    print(f"Status: {result.status.value}")
    print(f"Result: {result.result}")
    print(f"Steps:  {result.steps}")
    print(f"Time:   {result.elapsed_seconds:.4f}s")

    await manager.publish_feedback(result, score=0.95, text="Echo worker performed correctly")
    print(f"\n--- Feedback published (score=0.95) ---")

    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
