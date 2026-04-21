from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tensordict import TensorDictBase
from torchrl.trainers.trainers import TrainerHookBase


class SteeringHook(TrainerHookBase):
    """Hook that applies an in-place policy steering transform on a batch."""

    def __init__(
        self,
        transform: Callable[[TensorDictBase], TensorDictBase] | None = None,
        destination: str = "pre_optim_steps",
    ) -> None:
        self.transform = transform
        self.destination = destination

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        if self.transform is None:
            return batch
        return self.transform(batch)

    def register(self, trainer: Any, name: str = "steering_hook") -> None:
        trainer.register_op(self.destination, self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"destination": self.destination}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.destination = state_dict.get("destination", self.destination)


class ProbingHook(TrainerHookBase):
    """Hook that collects interpretability probes from batches during training."""

    def __init__(
        self,
        probe: Callable[[TensorDictBase], None] | None = None,
        destination: str = "post_steps",
    ) -> None:
        self.probe = probe
        self.destination = destination

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        if self.probe is not None:
            self.probe(batch)
        return batch

    def register(self, trainer: Any, name: str = "probing_hook") -> None:
        trainer.register_op(self.destination, self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"destination": self.destination}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.destination = state_dict.get("destination", self.destination)
