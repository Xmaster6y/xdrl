from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict
from torchrl.trainers.trainers import TrainerHookBase

from xdrl.trainer_hooks import (
    MultiAgentGAEHook,
    PolicyCheckpointHook,
    ReduceLossTensorsHook,
    WeightedSumRewardHook,
)


def _multi_agent_gae_hook() -> MultiAgentGAEHook:
    loss = MagicMock()
    loss.critic_network = torch.nn.Linear(4, 1)
    loss.critic_network_params = []
    loss.target_critic_network_params = []
    return MultiAgentGAEHook(loss_module=loss, gamma=0.99, lmbda=0.95, group="agents")


def test_multi_agent_gae_hook_runs_gae_when_per_group_next_keys():
    batch = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": torch.zeros(2, 3, 4, 5),
                },
                batch_size=[2, 3, 4],
            ),
            "next": TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "reward": torch.randn(2, 3, 4, 1),
                            "done": torch.zeros(2, 3, 4, 1, dtype=torch.bool),
                            "terminated": torch.zeros(2, 3, 4, 1, dtype=torch.bool),
                        },
                        batch_size=[2, 3, 4],
                    ),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = _multi_agent_gae_hook()
    hook.gae = MagicMock()
    hook(batch)
    hook.gae.assert_called_once()


def test_multi_agent_gae_hook_raises_when_only_shared_next_keys():
    batch = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": torch.zeros(2, 3, 4, 5),
                },
                batch_size=[2, 3, 4],
            ),
            "next": TensorDict(
                {
                    "reward": torch.randn(2, 3, 1),
                    "done": torch.zeros(2, 3, 1, dtype=torch.bool),
                    "terminated": torch.zeros(2, 3, 1, dtype=torch.bool),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = _multi_agent_gae_hook()
    with pytest.raises(RuntimeError, match="missing"):
        hook(batch)


def test_reduce_loss_tensors_reduces_non_scalars():
    losses = TensorDict(
        {
            "loss_objective": torch.tensor(1.0),
            "explained_variance": torch.ones(2, 3, 1),
        },
        batch_size=[],
    )

    hook = ReduceLossTensorsHook()
    hook(losses, losses)

    assert losses.get("loss_objective").shape == torch.Size([])
    assert losses.get("explained_variance").shape == torch.Size([])
    assert losses.get("explained_variance").item() == pytest.approx(1.0)


def test_marl_hooks_inherit_trainer_hook_base():
    assert issubclass(MultiAgentGAEHook, TrainerHookBase)
    assert issubclass(ReduceLossTensorsHook, TrainerHookBase)


def test_policy_checkpoint_hook_saves_periodically(tmp_path):
    policy = torch.nn.Linear(4, 2)
    hook = PolicyCheckpointHook(policy=policy, directory=tmp_path, interval=2, prefix="policy")
    batch = TensorDict({}, batch_size=[])

    hook(batch)
    assert list(tmp_path.glob("*.pt")) == []

    hook(batch)
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1

    checkpoint = torch.load(checkpoints[0], map_location="cpu")
    assert checkpoint["step"] == 2
    assert "policy_state_dict" in checkpoint
    assert checkpoint["meta"] == {}
    torch.testing.assert_close(checkpoint["policy_state_dict"]["weight"], policy.state_dict()["weight"])
    torch.testing.assert_close(checkpoint["policy_state_dict"]["bias"], policy.state_dict()["bias"])


def test_policy_checkpoint_hook_inherits_trainer_hook_base():
    assert issubclass(PolicyCheckpointHook, TrainerHookBase)


def test_weighted_sum_reward_default_weights():
    reward = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 1.0]])
    batch = TensorDict(
        {"next": TensorDict({"reward": reward}, batch_size=[2])},
        batch_size=[2],
    )
    hook = WeightedSumRewardHook(weights=None, preserve_input_key=None)
    hook(batch)

    scalar = batch.get(("next", "reward"))
    assert scalar.shape == torch.Size([2, 1])
    assert scalar.squeeze(-1).tolist() == pytest.approx([6.0, 2.0])


def test_weighted_sum_reward_with_custom_weights():
    reward = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    batch = TensorDict(
        {"next": TensorDict({"reward": reward}, batch_size=[1, 2])},
        batch_size=[1, 2],
    )
    hook = WeightedSumRewardHook(weights=[0.2, 0.8], preserve_input_key=None)
    hook(batch)

    scalar = batch.get(("next", "reward"))
    assert scalar.shape == torch.Size([1, 2, 1])
    torch.testing.assert_close(scalar.squeeze(-1), torch.tensor([[1.8, 3.8]]))


def test_weighted_sum_reward_hook_overwrites_and_preserves_vector_reward():
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "reward": torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
                },
                batch_size=[2],
            )
        },
        batch_size=[2],
    )
    hook = WeightedSumRewardHook(weights=[0.5, 0.5])

    hook(batch)

    assert batch.get(("next", "reward")).shape == torch.Size([2, 1])
    assert batch.get(("next", "reward")).squeeze(-1).tolist() == pytest.approx([1.5, 3.0])
    assert batch.get(("next", "reward_vector")).shape == torch.Size([2, 2])
