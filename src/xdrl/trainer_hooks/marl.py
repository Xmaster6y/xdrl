from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDictBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.trainers.trainers import TrainerHookBase


class MultiAgentGAEHook(TrainerHookBase):
    """Pre-epoch hook that computes GAE for MARL batches with per-group ``next`` keys."""

    def __init__(self, loss_module: ClipPPOLoss, gamma: float, lmbda: float, group: str = "agents") -> None:
        self.loss_module = loss_module
        self.group = group
        self.gae = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.loss_module.critic_network,
            average_gae=True,
        )
        self._set_gae_keys()

    def _set_gae_keys(self) -> None:
        group = self.group
        self.gae.set_keys(
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
        )

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        group = self.group
        keys = set(batch.keys(True, True))
        required = (
            ("next", group, "reward"),
            ("next", group, "done"),
            ("next", group, "terminated"),
        )
        missing = [k for k in required if k not in keys]
        if missing:
            missing_str = ", ".join(str(k) for k in missing)
            msg = f"MARL batch must define per-group keys under `next` for group {group!r}; missing: {missing_str}"
            raise RuntimeError(msg)
        with torch.no_grad():
            self.gae(
                batch,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )
        return batch

    def register(self, trainer: Any, name: str = "multiagent_gae") -> None:
        trainer.register_op("pre_epoch", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"group": self.group}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)
        self._set_gae_keys()


class ExpandSharedNextKeysHook(TrainerHookBase):
    """Expand shared ``next`` keys to per-group keys for MARL batches."""

    def __init__(self, group: str = "agents", key_names: tuple[str, ...] = ("done", "terminated")) -> None:
        self.group = group
        self.key_names = key_names

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        group = self.group
        if group not in batch.keys():
            return batch

        group_shape = batch.get(group).shape
        keys = set(batch.keys(True, True))
        for key_name in self.key_names:
            nested_key = ("next", group, key_name)
            shared_key = ("next", key_name)
            if nested_key in keys or shared_key not in keys:
                continue
            batch.set(
                nested_key,
                batch.get(shared_key).unsqueeze(-1).expand((*group_shape, 1)),
            )
        return batch

    def register(self, trainer: Any, name: str = "expand_shared_next_keys") -> None:
        trainer.register_op("pre_epoch", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"group": self.group, "key_names": self.key_names}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)
        self.key_names = tuple(state_dict.get("key_names", self.key_names))


class ReduceLossTensorsHook(TrainerHookBase):
    """Reduce non-scalar loss statistics to scalars for logging."""

    def __call__(self, _sub_batch: TensorDictBase, losses_td: TensorDictBase) -> TensorDictBase:
        for key, value in losses_td.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                losses_td.set(key, value.mean())
        return losses_td

    def register(self, trainer: Any, name: str = "reduce_loss_tensors") -> None:
        trainer.register_op("process_loss", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return None
