import pytest
import torch
from tensordict import TensorDict

from xdrl.training import ensure_group_next_keys, reduce_loss_tensors


def test_ensure_group_next_keys_broadcasts_shared_signals():
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

    ensure_group_next_keys(batch, group="agents")

    assert ("next", "agents", "reward") in batch.keys(True, True)
    assert ("next", "agents", "done") in batch.keys(True, True)
    assert ("next", "agents", "terminated") in batch.keys(True, True)
    assert batch.get(("next", "agents", "reward")).shape == torch.Size([2, 3, 4, 1])
    assert batch.get(("next", "agents", "done")).shape == torch.Size([2, 3, 4, 1])
    assert batch.get(("next", "agents", "terminated")).shape == torch.Size([2, 3, 4, 1])


def test_reduce_loss_tensors_reduces_non_scalars():
    losses = TensorDict(
        {
            "loss_objective": torch.tensor(1.0),
            "explained_variance": torch.ones(2, 3, 1),
        },
        batch_size=[],
    )

    reduce_loss_tensors(losses, losses)

    assert losses.get("loss_objective").shape == torch.Size([])
    assert losses.get("explained_variance").shape == torch.Size([])
    assert losses.get("explained_variance").item() == pytest.approx(1.0)
