"""OpenRLHFLauncher — exports datasets and launches OpenRLHF CLI as subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.events.types import TrainingRolloutEvent

logger = logging.getLogger(__name__)


def export_dataset(rollouts: list[TrainingRolloutEvent], path: Path) -> None:
    """Export rollouts to JSONL format for OpenRLHF consumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rollouts:
            record = {
                "prompt": r.prompt,
                "response": r.response,
                "reward": r.outcome_score,
                "step_scores": r.step_scores,
            }
            f.write(json.dumps(record) + "\n")
    logger.info("Exported %d rollouts to %s", len(rollouts), path)


class OpenRLHFLauncher:
    """Launches OpenRLHF GRPO training as a subprocess.

    Not a Trainer protocol implementation — separate orchestration class
    for production GPU training via Ray.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: str = "openrlhf_checkpoints",
    ) -> None:
        self._model_name = model_name
        self._output_dir = Path(output_dir)
        self._process: asyncio.subprocess.Process | None = None

    async def launch(
        self,
        dataset_path: Path,
        *,
        extra_args: list[str] | None = None,
    ) -> int:
        """Launch openrlhf.cli.train_grpo_ray as subprocess.

        Returns the process exit code.
        """
        cmd = [
            "python", "-m", "openrlhf.cli.train_grpo_ray",
            "--pretrain", self._model_name,
            "--dataset", str(dataset_path),
            "--save_path", str(self._output_dir),
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.info("Launching OpenRLHF: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await self._process.communicate()

        if self._process.returncode == 0:
            logger.info("OpenRLHF training completed successfully")
        else:
            logger.error(
                "OpenRLHF training failed (exit %d): %s",
                self._process.returncode,
                stderr.decode()[-500:] if stderr else "no stderr",
            )

        return self._process.returncode

    async def monitor_checkpoints(self) -> str | None:
        """Find the latest checkpoint directory in output_dir."""
        if not self._output_dir.exists():
            return None
        checkpoint_dirs = sorted(
            [d for d in self._output_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        if checkpoint_dirs:
            return str(checkpoint_dirs[-1])
        return None
