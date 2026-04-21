from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import LineariseRewards
from torchrl.trainers.trainers import TrainerHookBase


class WeightedSumRewardHook(TrainerHookBase):
    """Trainer hook that scalarizes a vector reward using a weighted sum."""

    def __init__(
        self,
        *,
        weights: Sequence[float] | None = None,
        in_key: str | tuple[str, ...] = ("next", "reward"),
        out_key: str | tuple[str, ...] = ("next", "reward"),
        preserve_input_key: str | tuple[str, ...] | None = ("next", "reward_vector"),
    ) -> None:
        self.weights = tuple(weights) if weights is not None else None
        self.in_key = in_key
        self.out_key = out_key
        self.preserve_input_key = preserve_input_key
        self._linearise = self._make_linearise()

    def _make_linearise(self) -> LineariseRewards:
        w = None if self.weights is None else torch.tensor(list(self.weights), dtype=torch.float32)
        return LineariseRewards(in_keys=["reward"], out_keys=["reward"], weights=w)

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        reward = batch.get(self.in_key)

        if self.preserve_input_key is not None and self.preserve_input_key not in batch.keys(True, True):
            batch.set(self.preserve_input_key, reward)

        if reward.ndim == 0:
            reward_scalar = reward.reshape(1)
        elif reward.shape[-1] == 1 and self.weights is None:
            reward_scalar = reward
        else:
            linearise = self._linearise.to(device=reward.device, dtype=reward.dtype)
            td = TensorDict({"reward": reward}, batch_size=reward.shape[:-1])
            reward_scalar = linearise(td)["reward"].unsqueeze(-1)

        batch.set(self.out_key, reward_scalar)
        return batch

    def register(self, trainer: Any, name: str = "weighted_sum_reward") -> None:
        trainer.register_op("pre_epoch", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights) if self.weights is not None else None,
            "in_key": self.in_key,
            "out_key": self.out_key,
            "preserve_input_key": self.preserve_input_key,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        weights = state_dict.get("weights")
        self.weights = tuple(weights) if weights is not None else None
        self.in_key = state_dict.get("in_key", self.in_key)
        self.out_key = state_dict.get("out_key", self.out_key)
        self.preserve_input_key = state_dict.get("preserve_input_key", self.preserve_input_key)
        self._linearise = self._make_linearise()
