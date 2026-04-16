from __future__ import annotations

import torch
from tensordict import TensorDictBase

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE


def ensure_group_next_keys(batch: TensorDictBase, group: str = "agents") -> None:
    """Broadcast shared next keys to per-agent keys when needed."""
    keys = set(batch.keys(True, True))

    group_reward_key = ("next", group, "reward")
    if group_reward_key in keys:
        reward = batch.get(group_reward_key)
        n_agents = reward.shape[-2]
    elif ("next", "reward") in keys:
        root_reward = batch.get(("next", "reward"))
        if group not in batch.keys():
            raise RuntimeError("Cannot infer number of agents for reward broadcast.")
        n_agents = batch.get(group).shape[-1]
        batch.set(
            group_reward_key,
            root_reward.unsqueeze(-2).expand(*root_reward.shape[:-1], n_agents, 1),
        )
    else:
        raise RuntimeError("Batch has neither ('next','agents','reward') nor ('next','reward').")

    if ("next", group, "done") not in keys and ("next", "done") in keys:
        root_done = batch.get(("next", "done"))
        batch.set(
            ("next", group, "done"),
            root_done.unsqueeze(-2).expand(*root_done.shape[:-1], n_agents, 1),
        )

    if ("next", group, "terminated") not in keys and ("next", "terminated") in keys:
        root_terminated = batch.get(("next", "terminated"))
        batch.set(
            ("next", group, "terminated"),
            root_terminated.unsqueeze(-2).expand(*root_terminated.shape[:-1], n_agents, 1),
        )


class MultiAgentGAEHook:
    """Pre-epoch hook that adapts VMAS batch keys and computes GAE."""

    def __init__(self, loss_module: ClipPPOLoss, gamma: float, lmbda: float, group: str = "agents") -> None:
        self.loss_module = loss_module
        self.group = group
        self.gae = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.loss_module.critic_network,
            average_gae=True,
        )
        self.gae.set_keys(
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
        )

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        ensure_group_next_keys(batch, self.group)
        with torch.no_grad():
            self.gae(
                batch,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )
        return batch


def reduce_loss_tensors(_sub_batch: TensorDictBase, losses_td: TensorDictBase) -> TensorDictBase:
    """Reduce non-scalar loss stats to scalars for logger compatibility."""
    for key, value in losses_td.items():
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            losses_td.set(key, value.mean())
    return losses_td
