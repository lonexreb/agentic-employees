"""Demo entry point: python -m src"""

import asyncio
import logging

from src.config import Config
from src.events.bus import EventBus
from src.events.topics import TRAINING_ROLLOUTS
from src.events.types import TrainingRolloutEvent
from src.inference.client import create_client
from src.manager.manager import Manager

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config()

    bus = EventBus()
    await bus.connect(cfg.nats_url)

    # Create shared inference client
    inference_client = create_client(
        backend=cfg.inference_backend,
        base_url=cfg.inference_base_url or cfg.ollama_host,
        api_key=cfg.inference_api_key,
    )
    print(f"Using inference backend: {cfg.inference_backend}")

    # Try LLMWorker with InferenceClient, else fall back to EchoWorker
    try:
        from src.workers.llm_worker import LLMWorker

        worker = LLMWorker(
            cfg.worker_id,
            bus,
            model=cfg.llm_model,
            client=inference_client,
        )
        print(f"Using LLMWorker with model={cfg.llm_model}")
    except (ImportError, Exception) as exc:
        logger.info("LLMWorker not available (%s), falling back to EchoWorker", exc)
        from src.workers.echo_worker import EchoWorker

        worker = EchoWorker(cfg.worker_id, bus)
        print("Using EchoWorker (fallback)")

    # Set up PRM evaluator
    try:
        from src.rewards.scorer import LLMJudgeScorer
        from src.rewards.prm_evaluator import PRMEvaluator

        scorer = LLMJudgeScorer(model=cfg.llm_model, client=inference_client)
        evaluator = PRMEvaluator(bus, scorer)
        await evaluator.start(["coding"])
        print("PRMEvaluator started")
    except (ImportError, Exception) as exc:
        logger.info("Could not start PRMEvaluator: %s", exc)
        evaluator = None

    # Optionally set up VLLMLoRAManager for hot-swap
    vllm_manager = None
    if cfg.inference_backend == "openai":
        try:
            from src.inference.vllm_lora import VLLMLoRAManager

            base = cfg.inference_base_url or "http://localhost:8000"
            vllm_manager = VLLMLoRAManager(base_url=base)
            models = await vllm_manager.health_check()
            print(f"VLLMLoRAManager connected, models: {models}")
        except Exception as exc:
            logger.info("VLLMLoRAManager not available: %s", exc)
            vllm_manager = None

    # Set up training loop
    trainer = None
    if cfg.trainer_backend == "openrlhf":
        from src.training.openrlhf_launcher import OpenRLHFLauncher

        _openrlhf = OpenRLHFLauncher(  # noqa: F841
            model_name=cfg.training_model,
            output_dir=cfg.training_checkpoint_dir,
        )
        print("OpenRLHFLauncher configured (will launch on batch)")
    else:
        try:
            from src.training.trainer import GRPOTrainer

            trainer = GRPOTrainer(
                model_name=cfg.training_model,
                lr=cfg.training_lr,
                clip_epsilon=cfg.training_clip_epsilon,
                kl_beta=cfg.training_kl_beta,
                checkpoint_dir=cfg.training_checkpoint_dir,
                lora_rank=cfg.training_lora_rank,
                device=cfg.training_device,
            )
            print(f"Using GRPOTrainer with model={cfg.training_model}")
        except (ImportError, Exception) as exc:
            logger.info("torch/transformers/peft not available (%s), using MockTrainer", exc)
            from src.training.trainer import MockTrainer

            trainer = MockTrainer()
            print("Using MockTrainer (fallback)")

    if trainer is not None:
        from src.training.bridge import RolloutBuffer
        from src.training.loop import TrainingLoop

        buffer = RolloutBuffer(
            batch_size=cfg.training_batch_size,
            group_size=cfg.training_group_size,
        )
        training_loop = TrainingLoop(bus, trainer, buffer)
        await training_loop.start()
        print("TrainingLoop started")

    # Capture rollouts for display
    rollout_event = asyncio.Event()
    captured_rollout: list[TrainingRolloutEvent] = []

    async def _on_rollout(rollout: TrainingRolloutEvent) -> None:
        captured_rollout.append(rollout)
        rollout_event.set()

    await bus.subscribe(TRAINING_ROLLOUTS, TrainingRolloutEvent, _on_rollout)

    manager = Manager(cfg.manager_id, bus)
    await manager.start()
    await worker.start()

    print("\n--- Assigning task ---")
    task = await manager.assign_task("coding", "Write a fibonacci function in Python")
    print(f"Task ID: {task.task_id}")

    result = await manager.wait_for_result(task.task_id, timeout=cfg.task_timeout_seconds)
    print("\n--- Result ---")
    print(f"Status: {result.status.value}")
    print(f"Result: {result.result[:200]}")
    print(f"Steps:  {result.steps}")
    print(f"Time:   {result.elapsed_seconds:.4f}s")
    if result.model_version:
        print(f"Model:  {result.model_version}")

    # Wait briefly for PRM scoring
    if evaluator:
        try:
            await asyncio.wait_for(rollout_event.wait(), timeout=30.0)
            rollout = captured_rollout[0]
            print("\n--- PRM Scores ---")
            for i, (step, score) in enumerate(
                zip(rollout.steps, rollout.step_scores), 1
            ):
                print(f"  Step {i}: {score:.3f} — {step[:80]}")
            print(f"  Outcome: {rollout.outcome_score:.3f}")
        except asyncio.TimeoutError:
            print("\n--- PRM scoring timed out ---")

    await manager.publish_feedback(result, score=0.95, text="Task completed")
    print("\n--- Feedback published (score=0.95) ---")

    if vllm_manager:
        await vllm_manager.close()
    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
