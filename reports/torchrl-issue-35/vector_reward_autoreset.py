"""Reproduce scalar/vector reward mixing after a Gymnasium autoreset."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv


def make_env(_name: str, **_kwargs: object) -> SimpleNamespace:
    env = SimpleNamespace(
        observation_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        reward_space=spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ),
        render_mode=None,
        metadata={"render_modes": []},
    )
    env.unwrapped = env
    env.reset = lambda *, seed=None, options=None: (np.zeros(1, dtype=np.float32), {})

    def step(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        action_array = np.asarray(action, dtype=np.float32)
        batch_shape = action_array.shape[:-1]
        return (
            np.zeros((*batch_shape, 1), dtype=np.float32),
            np.ones((*batch_shape, 2), dtype=np.float32),
            action_array[..., 0] > 0.5,
            np.zeros(batch_shape, dtype=bool),
            {},
        )

    env.step = step
    env.close = lambda: None
    return env


def main() -> None:
    with patch.object(GymEnv, "lib", new_callable=PropertyMock, return_value=SimpleNamespace(make=make_env)):
        env = GymEnv("vector-reward-autoreset", num_envs=2)
    try:
        transition = env.reset()
        transition.set("action", torch.tensor([[1.0], [0.0]], dtype=torch.float32))
        transition = env.step(transition)
        assert transition.get(("next", "done")).squeeze(-1).tolist() == [True, False]
        env.step(TensorDict({"action": torch.zeros(2, 1)}, batch_size=transition.batch_size))
    finally:
        env.close()


if __name__ == "__main__":
    main()
