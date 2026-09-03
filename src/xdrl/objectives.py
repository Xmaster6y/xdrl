"""Algorithm-aware views over native TorchRL loss modules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TypeAlias

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.objectives import DQNLoss, IQLLoss, LossModule, PPOLoss, SACLoss
from torchrl.objectives.multiagent import QMixerLoss

from xdrl.interpretation import Component


@dataclass(frozen=True, slots=True)
class QValueTargets:
    """Target-evaluation members of one Q-value ensemble."""

    qvalue: tuple[Component, ...]


@dataclass(frozen=True, slots=True)
class DQNInterpretation:
    """Online and delayed Q-value components of a native DQN loss."""

    loss: DQNLoss
    qvalue: tuple[Component, ...]
    target: QValueTargets


@dataclass(frozen=True, slots=True)
class PPOInterpretation:
    """Actor and critic components of a native PPO loss."""

    loss: PPOLoss
    actor: Component
    critic: Component | None

    @property
    def value(self) -> Component | None:
        """The PPO critic's state-value interpretation."""

        return self.critic


@dataclass(frozen=True, slots=True)
class SACTargets:
    """Delayed actor, Q-value, and optional value components of SAC."""

    actor: Component | None
    qvalue: tuple[Component, ...]
    value: Component | None


@dataclass(frozen=True, slots=True)
class SACInterpretation:
    """Actor and value components encoded by a native SAC loss."""

    loss: SACLoss
    actor: Component
    qvalue: tuple[Component, ...]
    value: Component | None
    target: SACTargets


@dataclass(frozen=True, slots=True)
class IQLInterpretation:
    """Actor, Q-value ensemble, and state value encoded by IQL."""

    loss: IQLLoss
    actor: Component
    qvalue: tuple[Component, ...]
    value: Component
    target: QValueTargets


@dataclass(frozen=True, slots=True)
class QMixerTargets:
    """Delayed local-value and mixing components of QMixer."""

    local_qvalue: Component | None
    mixer: Component | None
    joint_qvalue: Component | None


@dataclass(frozen=True, slots=True)
class QMixerInterpretation:
    """Local, mixing, and joint value computations encoded by QMixer."""

    loss: QMixerLoss
    local_qvalue: Component
    mixer: Component
    joint_qvalue: Component
    target: QMixerTargets


ObjectiveInterpretation: TypeAlias = (
    DQNInterpretation | PPOInterpretation | SACInterpretation | IQLInterpretation | QMixerInterpretation
)


@singledispatch
def interpret_objective(loss: LossModule) -> ObjectiveInterpretation:
    """Interpret one explicitly supported TorchRL objective."""

    raise TypeError(
        f"XDRL has no objective integration for {type(loss).__name__}; "
        "register an explicit interpret_objective adapter for this TorchRL loss"
    )


@interpret_objective.register
def _interpret_dqn(loss: DQNLoss) -> ObjectiveInterpretation:
    online = Component(loss.value_network, "qvalue", loss.value_network_params)
    target = ()
    if loss.delay_value:
        target = (Component(loss.value_network, "target.qvalue", loss.target_value_network_params),)
    return DQNInterpretation(loss=loss, qvalue=(online,), target=QValueTargets(qvalue=target))


@interpret_objective.register
def _interpret_ppo(loss: PPOLoss) -> ObjectiveInterpretation:
    actor = Component(loss.actor_network, "actor", loss.actor_network_params)
    critic = None
    if loss.critic_network is not None:
        critic = Component(loss.critic_network, "critic", loss.critic_network_params)
    return PPOInterpretation(loss=loss, actor=actor, critic=critic)


@interpret_objective.register
def _interpret_sac(loss: SACLoss) -> ObjectiveInterpretation:
    qvalue = _ensemble(loss.qvalue_network, "qvalue", loss.qvalue_network_params, loss.num_qvalue_nets)
    target_qvalue = ()
    if loss.delay_qvalue:
        target_qvalue = _ensemble(
            loss.qvalue_network,
            "target.qvalue",
            loss.target_qvalue_network_params,
            loss.num_qvalue_nets,
        )
    value = None
    target_value = None
    value_network = getattr(loss, "value_network", None)
    if value_network is not None:
        value = Component(value_network, "value", loss.value_network_params)
        if loss.delay_value:
            target_value = Component(value_network, "target.value", loss.target_value_network_params)
    target_actor = None
    if loss.delay_actor:
        target_actor = Component(loss.actor_network, "target.actor", loss.target_actor_network_params)
    return SACInterpretation(
        loss=loss,
        actor=Component(loss.actor_network, "actor", loss.actor_network_params),
        qvalue=qvalue,
        value=value,
        target=SACTargets(actor=target_actor, qvalue=target_qvalue, value=target_value),
    )


@interpret_objective.register
def _interpret_iql(loss: IQLLoss) -> ObjectiveInterpretation:
    count = loss.num_qvalue_nets
    return IQLInterpretation(
        loss=loss,
        actor=Component(loss.actor_network, "actor", loss.actor_network_params),
        qvalue=_ensemble(loss.qvalue_network, "qvalue", loss.qvalue_network_params, count),
        value=Component(loss.value_network, "value", loss.value_network_params),
        target=QValueTargets(
            qvalue=_ensemble(loss.qvalue_network, "target.qvalue", loss.target_qvalue_network_params, count)
        ),
    )


@interpret_objective.register
def _interpret_qmixer(loss: QMixerLoss) -> ObjectiveInterpretation:
    online_joint_params = _joint_params(loss.local_value_network_params, loss.mixer_network_params)
    local = Component(loss.local_value_network, "local_qvalue", loss.local_value_network_params)
    mixer = Component(loss.mixer_network, "mixer", loss.mixer_network_params)
    joint = Component(loss.global_value_network, "joint_qvalue", online_joint_params)
    target = QMixerTargets(local_qvalue=None, mixer=None, joint_qvalue=None)
    if loss.delay_value:
        target_local = Component(loss.local_value_network, "target.local_qvalue", loss.target_local_value_network_params)
        target_mixer = Component(loss.mixer_network, "target.mixer", loss.target_mixer_network_params)
        target_joint = Component(
            loss.global_value_network,
            "target.joint_qvalue",
            _joint_params(loss.target_local_value_network_params, loss.target_mixer_network_params),
        )
        target = QMixerTargets(
            local_qvalue=target_local,
            mixer=target_mixer,
            joint_qvalue=target_joint,
        )
    return QMixerInterpretation(
        loss=loss,
        local_qvalue=local,
        mixer=mixer,
        joint_qvalue=joint,
        target=target,
    )


def _ensemble(module: TensorDictModuleBase, name: str, params: TensorDict, count: int) -> tuple[Component, ...]:
    if count == 1 or params.ndim == 0:
        return (Component(module, name, params),)
    return tuple(Component(module, f"{name}[{index}]", params[index]) for index in range(count))


def _joint_params(local_params: TensorDict, mixer_params: TensorDict) -> TensorDict:
    return TensorDict(
        {"module": {"0": local_params, "1": mixer_params}},
        batch_size=local_params.batch_size,
        device=local_params.device,
    )


__all__ = [
    "DQNInterpretation",
    "IQLInterpretation",
    "ObjectiveInterpretation",
    "PPOInterpretation",
    "QMixerInterpretation",
    "QMixerTargets",
    "QValueTargets",
    "SACInterpretation",
    "SACTargets",
    "interpret_objective",
]
