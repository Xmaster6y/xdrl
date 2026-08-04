"""Reproduce legacy Gym/CV2 imports from Gymnasium pixel detection."""

import sys
from types import SimpleNamespace

import gymnasium
import numpy as np
from torchrl.envs.libs.gym import _is_from_pixels, set_gym_backend


set_gym_backend(gymnasium).set()
assert "gym" not in sys.modules
assert "cv2" not in sys.modules
env = SimpleNamespace(observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
_is_from_pixels(env)
unexpected = [name for name in ("gym", "cv2") if name in sys.modules]
assert not unexpected, f"unexpected imports: {unexpected}"
