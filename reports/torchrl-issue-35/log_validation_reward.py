"""Reproduce LogValidationReward closing a reusable evaluation environment."""

from types import SimpleNamespace

import torch
from gymnasium.error import ClosedEnvironmentError
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType
from torchrl.trainers.trainers import LogValidationReward


class ValidationEnv:
    def __init__(self) -> None:
        self.closed = False
        self.transform = SimpleNamespace(dump=lambda suffix=None: None)

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def rollout(self, **_kwargs: object) -> TensorDict:
        if self.closed:
            raise ClosedEnvironmentError("Trying to operate on a closed eval env.")
        return TensorDict(
            {
                "done": torch.zeros(3, 1, dtype=torch.bool),
                "next": TensorDict(
                    {"done": torch.zeros(3, 1, dtype=torch.bool), "reward": torch.ones(3, 1)}, batch_size=[3]
                ),
            },
            batch_size=[3],
        )

    def close(self) -> None:
        self.closed = True

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, _state_dict: dict[str, object]) -> None:
        pass


hook = LogValidationReward(
    record_interval=1,
    record_frames=3,
    frame_skip=1,
    policy_exploration=torch.nn.Identity(),
    environment=ValidationEnv(),
    exploration_type=ExplorationType.DETERMINISTIC,
    log_keys=[("next", "reward")],
)
hook(TensorDict({}, batch_size=[]))
hook(TensorDict({}, batch_size=[]))
