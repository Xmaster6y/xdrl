import torch
from tensordict import TensorDict

from xdrl.trainer import _process_batch_for_qmix


def _qmix_batch() -> TensorDict:
    done = torch.tensor(
        [
            [[[False], [True], [False]], [[False], [False], [False]]],
            [[[True], [True], [False]], [[False], [False], [True]]],
        ]
    )
    terminated = torch.tensor(
        [
            [[[False], [False], [False]], [[True], [False], [False]]],
            [[[False], [True], [False]], [[False], [False], [False]]],
        ]
    )
    reward = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3, 1)

    return TensorDict(
        {
            "agents": TensorDict({"observation": torch.zeros(2, 2, 3, 5)}, batch_size=[2, 2, 3]),
            "next": TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "reward": reward,
                            "done": done,
                            "terminated": terminated,
                        },
                        batch_size=[2, 2, 3],
                    ),
                },
                batch_size=[2, 2],
            ),
        },
        batch_size=[2, 2],
    )


def test_process_batch_for_qmix_populates_shared_keys_from_group_keys() -> None:
    batch = _qmix_batch()

    out = _process_batch_for_qmix(batch, group="agents")

    expected_done = batch.get(("next", "agents", "done")).any(dim=-2)
    expected_terminated = batch.get(("next", "agents", "terminated")).any(dim=-2)
    expected_reward = batch.get(("next", "agents", "reward")).mean(dim=-2)

    torch.testing.assert_close(out.get(("next", "done")), expected_done)
    torch.testing.assert_close(out.get(("next", "terminated")), expected_terminated)
    torch.testing.assert_close(out.get(("next", "reward")), expected_reward)


def test_process_batch_for_qmix_does_not_override_existing_shared_keys() -> None:
    batch = _qmix_batch()
    existing_reward = torch.full((2, 2, 1), 123.0)
    existing_done = torch.ones((2, 2, 1), dtype=torch.bool)
    existing_terminated = torch.zeros((2, 2, 1), dtype=torch.bool)
    batch.set(("next", "reward"), existing_reward.clone())
    batch.set(("next", "done"), existing_done.clone())
    batch.set(("next", "terminated"), existing_terminated.clone())

    out = _process_batch_for_qmix(batch, group="agents")

    torch.testing.assert_close(out.get(("next", "reward")), existing_reward)
    torch.testing.assert_close(out.get(("next", "done")), existing_done)
    torch.testing.assert_close(out.get(("next", "terminated")), existing_terminated)
