from __future__ import annotations

from hydra.core.config_store import ConfigStore

from xdrl.configs.hooks import (
    GAEHookConfig,
    LogValidationRewardHookConfig,
    LoggingHookSetConfig,
    PolicyCheckpointHookConfig,
    WandbFinishHookConfig,
    WandbFlushHookConfig,
)

_REGISTERED = False


def register_configs() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(group="xdrl_hook", name="logging", node=LoggingHookSetConfig)
    cs.store(group="xdrl_hook", name="validation_reward", node=LogValidationRewardHookConfig)
    cs.store(group="xdrl_hook", name="gae", node=GAEHookConfig)
    cs.store(group="xdrl_hook", name="policy_checkpoint", node=PolicyCheckpointHookConfig)
    cs.store(group="xdrl_hook", name="wandb_finish", node=WandbFinishHookConfig)
    cs.store(group="xdrl_hook", name="wandb_flush", node=WandbFlushHookConfig)

    _REGISTERED = True


__all__ = ["register_configs"]
