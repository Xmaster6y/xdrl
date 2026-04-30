from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchrl.trainers.trainers import TrainerHookBase, _resolve_module


class PolicyCheckpointHook(TrainerHookBase):
    """Periodically checkpoint policy weights for offline analysis.

    Args:
        directory: Directory where ``.pt`` checkpoint files are written.
        interval: Number of hook calls between checkpoints. Non-positive values
            disable file writes while still allowing the hook to be registered.
        policy: Policy module to checkpoint. If omitted, ``policy_path`` is
            resolved from the trainer during registration.
        policy_path: Dotted path to the policy module on the trainer.
        prefix: Filename prefix used for checkpoint files.
        destination: TorchRL trainer operation where the hook is registered.
        meta: Extra metadata persisted in every checkpoint payload.
    """

    def __init__(
        self,
        *,
        directory: str | Path,
        interval: int,
        policy: torch.nn.Module | None = None,
        policy_path: str | None = None,
        prefix: str = "policy",
        destination: str = "post_steps",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.policy_path = policy_path
        self.directory = Path(directory)
        self.interval = int(interval)
        self.prefix = prefix
        self.destination = destination
        self.meta = {} if meta is None else dict(meta)
        self.num_calls = 0
        self.last_checkpoint_path: Path | None = None

    def __call__(self) -> None:
        if self.interval <= 0:
            return
        if self.policy is None:
            msg = "PolicyCheckpointHook has no policy. Set policy or policy_path."
            raise RuntimeError(msg)

        self.num_calls += 1
        if self.num_calls % self.interval != 0:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.directory / f"{self.prefix}_step_{self.num_calls:08d}.pt"
        payload = {
            "policy_state_dict": self.policy.state_dict(),
            "step": self.num_calls,
            "meta": dict(self.meta),
        }
        torch.save(payload, checkpoint_path)
        self.last_checkpoint_path = checkpoint_path

    def register(self, trainer: Any, name: str = "policy_checkpoint_hook") -> None:
        trainer.register_module(name, self)
        if self.interval <= 0:
            return
        if self.policy is None and self.policy_path is not None:
            self.policy = _resolve_module(trainer, self.policy_path)
        trainer.register_op(self.destination, self)

    def state_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "interval": self.interval,
            "prefix": self.prefix,
            "destination": self.destination,
            "policy_path": self.policy_path,
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
        self.policy_path = state_dict.get("policy_path", self.policy_path)
        self.meta = dict(state_dict.get("meta", self.meta))
        self.num_calls = int(state_dict.get("num_calls", self.num_calls))
        last_path = state_dict.get("last_checkpoint_path", None)
        self.last_checkpoint_path = None if last_path is None else Path(last_path)
