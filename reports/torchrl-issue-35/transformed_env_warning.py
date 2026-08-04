"""Reproduce the obsolete TransformedEnv auto-unwrap warning target."""

import warnings
from unittest.mock import patch

from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv


inner = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
outer = None
try:
    with patch("torchrl._utils._AUTO_UNWRAP", None):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            outer = TransformedEnv(inner, StepCounter())
    messages = [str(warning.message) for warning in captured]
    assert messages, "expected the auto-unwrap warning"
    assert all("0.9" not in message for message in messages), messages
finally:
    (outer or inner).close()
