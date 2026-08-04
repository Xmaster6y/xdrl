"""Reproduce continuous SAC config forwarding discrete-only parameters."""

from unittest.mock import patch

from hydra.utils import instantiate
from torchrl.trainers.algorithms.configs import objectives
from torchrl.trainers.algorithms.configs.objectives import SACLossConfig


captured_kwargs: dict[str, object] | None = None


class FakeSACLoss:
    pass


def capture_sac_loss(*_args: object, **kwargs: object) -> FakeSACLoss:
    global captured_kwargs
    captured_kwargs = kwargs
    return FakeSACLoss()


with patch.object(objectives, "SACLoss", capture_sac_loss):
    instantiate(
        SACLossConfig(
            actor_network=object(),
            qvalue_network=object(),
            discrete=False,
            action_space="categorical",
            num_actions=2,
            target_entropy_weight=0.98,
        )
    )

assert captured_kwargs is not None
unexpected = {"action_space", "num_actions", "target_entropy_weight"} & captured_kwargs.keys()
assert not unexpected, f"continuous SACLoss received discrete-only fields: {sorted(unexpected)}"
