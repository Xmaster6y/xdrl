from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDictBase
from torchrl.trainers.trainers import TrainerHookBase


class PolicyCheckpointHook(TrainerHookBase):
    """Periodically checkpoint policy weights for offline analysis."""

    def __init__(
        self,
        policy: torch.nn.Module,
        directory: str | Path,
        interval: int,
        prefix: str = "policy",
        destination: str = "post_steps",
        meta: dict[str, Any] | None = None,
    ) -> None:
        if interval <= 0:
            msg = "interval must be a positive integer"
            raise ValueError(msg)

        self.policy = policy
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.prefix = prefix
        self.destination = destination
        self.meta = {} if meta is None else dict(meta)
        self.num_calls = 0
        self.last_checkpoint_path: Path | None = None

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        self.num_calls += 1
        if self.num_calls % self.interval != 0:
            return batch

        checkpoint_path = self.directory / f"{self.prefix}_step_{self.num_calls:08d}.pt"
        payload = {
            "policy_state_dict": self.policy.state_dict(),
            "step": self.num_calls,
            "meta": dict(self.meta),
        }
        torch.save(payload, checkpoint_path)
        self.last_checkpoint_path = checkpoint_path
        return batch

    def register(self, trainer: Any, name: str = "policy_checkpoint_hook") -> None:
        trainer.register_op(self.destination, self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "interval": self.interval,
            "prefix": self.prefix,
            "destination": self.destination,
            "meta": dict(self.meta),
            "num_calls": self.num_calls,
            "last_checkpoint_path": None if self.last_checkpoint_path is None else str(self.last_checkpoint_path),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.directory = Path(state_dict.get("directory", self.directory))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = int(state_dict.get("interval", self.interval))
        self.prefix = state_dict.get("prefix", self.prefix)
        self.destination = state_dict.get("destination", self.destination)
        self.meta = dict(state_dict.get("meta", self.meta))
        self.num_calls = int(state_dict.get("num_calls", self.num_calls))
        last_path = state_dict.get("last_checkpoint_path", None)
        self.last_checkpoint_path = None if last_path is None else Path(last_path)
