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
    auto_unwrap_warnings = [
        warning
        for warning in captured
        if issubclass(warning.category, UserWarning) and "automatically unwrapped" in str(warning.message)
    ]
    assert len(auto_unwrap_warnings) == 1, "expected exactly one auto-unwrap warning"
    assert "0.9" not in str(auto_unwrap_warnings[0].message)
finally:
    (outer or inner).close()
