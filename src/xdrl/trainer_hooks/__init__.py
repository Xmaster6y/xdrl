from xdrl.trainer_hooks.marl import ExpandSharedNextKeysHook, MultiAgentGAEHook, ReduceLossTensorsHook
from xdrl.trainer_hooks.morl import WeightedSumRewardHook
from xdrl.trainer_hooks.checkpoints import PolicyCheckpointHook
from xdrl.trainer_hooks.logging import (
    LoggingCollectionMetricsHook,
    LoggingCountersHook,
    LoggingEvaluationHookSet,
    LoggingEvaluationMetricsHook,
    LoggingHookSet,
)

__all__ = [
    "MultiAgentGAEHook",
    "ExpandSharedNextKeysHook",
    "ReduceLossTensorsHook",
    "WeightedSumRewardHook",
    "PolicyCheckpointHook",
    "LoggingCollectionMetricsHook",
    "LoggingCountersHook",
    "LoggingEvaluationHookSet",
    "LoggingEvaluationMetricsHook",
    "LoggingHookSet",
]
