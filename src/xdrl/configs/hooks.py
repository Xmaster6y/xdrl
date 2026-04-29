from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict
from tensordict import TensorDictBase
from torchrl.trainers.algorithms.configs.common import ConfigBase, _normalize_hydra_key
from torchrl.objectives.value import GAE
from torchrl.trainers.trainers import LogValidationReward, Trainer, TrainerHookBase

from xdrl.trainer_hooks import LoggingHookSet, PolicyCheckpointHook
from xdrl.trainer_hooks.checkpoints import _resolve_attr_path


class ReducedLogValidationReward(LogValidationReward):
    def __init__(
        self,
        *,
        enabled: bool = True,
        pre_eval: bool = False,
        policy_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.enabled = bool(enabled)
        self.pre_eval = bool(pre_eval)
        self.policy_path = policy_path
        self._trainer: Trainer | None = None

    def _run_pre_eval(self) -> None:
        if self._trainer is None:
            return
        metrics = self(TensorDict({}, batch_size=[]))
        if metrics:
            self._trainer._log(**metrics)

    @torch.inference_mode()
    def __call__(self, batch: TensorDictBase) -> dict[str, Any]:
        if not self.enabled:
            return {}
        out = super().__call__(batch)
        if out is None:
            return {}
        reduced = {}
        for key, value in out.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                reduced[key] = value.float().mean()
            else:
                reduced[key] = value
        return reduced

    def register(self, trainer: Trainer, name: str = "validation_reward") -> None:
        self._trainer = trainer
        trainer.register_module(name, self)
        if not self.enabled:
            return
        if self.policy_exploration is None and self.policy_path is not None:
            self.policy_exploration = _resolve_attr_path(trainer, self.policy_path)
        if self.pre_eval:
            trainer.register_op("setup", self._run_pre_eval)
        trainer.register_op("post_steps_log", self)

    def state_dict(self) -> dict[str, Any]:
        state = {}
        if self.environment is not None:
            state.update(super().state_dict())
        state.update(
            {
                "enabled": self.enabled,
                "pre_eval": self.pre_eval,
                "policy_path": self.policy_path,
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.pre_eval = bool(state_dict.get("pre_eval", self.pre_eval))
        self.policy_path = state_dict.get("policy_path", self.policy_path)
        if self.environment is not None and "recorder_state_dict" in state_dict:
            super().load_state_dict(state_dict)


class GAEHook(torch.nn.Module, TrainerHookBase):
    def __init__(
        self,
        *,
        gamma: float,
        lmbda: float,
        value_network_path: str = "loss_module.critic_network",
        reward_key: Any = "reward",
        done_key: Any = "done",
        terminated_key: Any = "terminated",
        advantage_key: Any = "advantage",
        value_target_key: Any = "value_target",
        value_key: Any = "state_value",
        average_gae: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.lmbda = float(lmbda)
        self.value_network_path = value_network_path
        self.reward_key = _normalize_hydra_key(reward_key)
        self.done_key = _normalize_hydra_key(done_key)
        self.terminated_key = _normalize_hydra_key(terminated_key)
        self.advantage_key = _normalize_hydra_key(advantage_key)
        self.value_target_key = _normalize_hydra_key(value_target_key)
        self.value_key = _normalize_hydra_key(value_key)
        self.average_gae = bool(average_gae)
        self.gae: GAE | None = None

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        if self.gae is None:
            msg = "GAEHook must be registered before it is called."
            raise RuntimeError(msg)
        with torch.no_grad():
            return self.gae(batch)

    def register(self, trainer: Trainer, name: str = "gae_hook") -> None:
        value_network = _resolve_attr_path(trainer, self.value_network_path)
        self.gae = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=value_network,
            average_gae=self.average_gae,
        )
        self.gae.set_keys(
            reward=self.reward_key,
            done=self.done_key,
            terminated=self.terminated_key,
            advantage=self.advantage_key,
            value_target=self.value_target_key,
            value=self.value_key,
        )
        trainer.register_module(name, self)
        trainer.register_op("pre_epoch", self)


def _make_logging_hook_set(
    *,
    group: str,
    frame_skip: int,
    reward_key: Any | None = None,
    done_key: Any = ("next", "done"),
    episode_reward_key: Any | None = None,
    episode_reward_weights: list[float] | None = None,
    reduce_stats: bool | None = None,
    eval_hook_set: LoggingEvaluationHookSet | None = None,
) -> LoggingHookSet:
    normalized_reward_key = None if reward_key is None else _normalize_hydra_key(reward_key)
    normalized_done_key = _normalize_hydra_key(done_key)
    normalized_episode_reward_key = None if episode_reward_key is None else _normalize_hydra_key(episode_reward_key)
    return LoggingHookSet(
        group=group,
        frame_skip=frame_skip,
        reward_key=normalized_reward_key,
        done_key=normalized_done_key,
        episode_reward_key=normalized_episode_reward_key,
        episode_reward_weights=episode_reward_weights,
        reduce_stats=reduce_stats,
        eval_hook_set=eval_hook_set,
    )


def _make_policy_checkpoint_hook(
    *,
    directory: str,
    interval: int,
    policy: Any = None,
    policy_path: str | None = None,
    prefix: str = "policy",
    destination: str = "post_steps",
    meta: dict[str, Any] | None = None,
) -> PolicyCheckpointHook:
    return PolicyCheckpointHook(
        directory=directory,
        interval=interval,
        policy=policy,
        policy_path=policy_path,
        prefix=prefix,
        destination=destination,
        meta=meta,
    )


def _make_log_validation_reward_hook(
    *,
    environment: Any,
    interval_frames: int,
    frames_per_batch: int,
    record_frames: int,
    policy_exploration: Any = None,
    policy_path: str | None = None,
    enabled: bool = True,
    pre_eval: bool = False,
    frame_skip: int = 1,
    exploration_type: str = "DETERMINISTIC",
    log_keys: list[Any] | None = None,
    out_keys: dict[Any, str] | None = None,
    log_pbar: bool = False,
) -> LogValidationReward:
    from torchrl.envs import ExplorationType

    record_interval = max(1, int(interval_frames) // max(1, int(frames_per_batch) * int(frame_skip)))
    exploration = getattr(ExplorationType, exploration_type.upper())

    normalized_log_keys = None
    if log_keys is not None:
        normalized_log_keys = [_normalize_hydra_key(key) for key in log_keys]

    normalized_out_keys = None
    if out_keys is not None:
        normalized_out_keys = {_normalize_hydra_key(key): value for key, value in out_keys.items()}

    if normalized_log_keys is None:
        normalized_log_keys = [("next", "agents", "reward")]

    if normalized_out_keys is None:
        normalized_out_keys = {}
        for key in normalized_log_keys:
            if key == ("next", "reward"):
                normalized_out_keys[key] = "r_evaluation"
            elif isinstance(key, tuple):
                normalized_out_keys[key] = "validation/" + "/".join(str(item) for item in key)
            else:
                normalized_out_keys[key] = f"validation/{key}"

    return ReducedLogValidationReward(
        enabled=enabled,
        pre_eval=pre_eval,
        policy_path=policy_path,
        record_interval=record_interval,
        record_frames=record_frames,
        frame_skip=frame_skip,
        policy_exploration=policy_exploration,
        environment=environment,
        exploration_type=exploration,
        log_keys=normalized_log_keys,
        out_keys=normalized_out_keys,
        log_pbar=log_pbar,
    )


@dataclass
class LoggingHookSetConfig(ConfigBase):
    group: str = "agents"
    frame_skip: int = 1
    reward_key: Any | None = None
    done_key: Any = ("next", "done")
    episode_reward_key: Any | None = None
    episode_reward_weights: list[float] | None = None
    reduce_stats: bool | None = None
    eval_hook_set: Any = None

    _target_: str = "xdrl.configs.hooks._make_logging_hook_set"

    def __post_init__(self) -> None:
        pass


@dataclass
class PolicyCheckpointHookConfig(ConfigBase):
    policy: Any = None
    policy_path: str | None = None
    directory: str = "checkpoints/policy"
    interval: int = 0
    prefix: str = "policy"
    destination: str = "post_steps"
    meta: dict[str, Any] | None = field(default_factory=dict)

    _target_: str = "xdrl.configs.hooks._make_policy_checkpoint_hook"

    def __post_init__(self) -> None:
        pass


@dataclass
class GAEHookConfig(ConfigBase):
    gamma: float = 0.99
    lmbda: float = 0.95
    value_network_path: str = "loss_module.critic_network"
    reward_key: Any = "reward"
    done_key: Any = "done"
    terminated_key: Any = "terminated"
    advantage_key: Any = "advantage"
    value_target_key: Any = "value_target"
    value_key: Any = "state_value"
    average_gae: bool = True

    _target_: str = "xdrl.configs.hooks.GAEHook"

    def __post_init__(self) -> None:
        pass


@dataclass
class LogValidationRewardHookConfig(ConfigBase):
    policy_exploration: Any = None
    policy_path: str | None = None
    environment: Any = None
    enabled: bool = True
    pre_eval: bool = False
    interval_frames: int = 100_000
    frames_per_batch: int = 1
    record_frames: int = 100
    frame_skip: int = 1
    exploration_type: str = "DETERMINISTIC"
    log_keys: list[Any] | None = field(default_factory=lambda: [["next", "agents", "reward"]])
    out_keys: dict[Any, str] | None = None
    log_pbar: bool = False

    _target_: str = "xdrl.configs.hooks._make_log_validation_reward_hook"

    def __post_init__(self) -> None:
        pass


@dataclass
class WandbFinishHookConfig(ConfigBase):
    enabled: bool = True

    _target_: str = "xdrl.trainer_hooks.logging.WandbFinishHook"

    def __post_init__(self) -> None:
        pass
