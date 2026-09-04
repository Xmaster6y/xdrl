"""Semantic views inferred from native TorchRL module classes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

from tdhook.workflow import Workflow, WorkflowResult
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.modules import ActorCriticOperator, ActorValueOperator, ProbabilisticActor, QValueActor, ValueOperator

from xdrl.interpretation import Component


@dataclass(frozen=True, slots=True)
class ModuleInterpretation:
    """RL functions already expressed by one TorchRL module."""

    module: TensorDictModuleBase
    actor: Component | None = None
    policy: Component | None = None
    critic: Component | None = None
    value: Component | None = None
    qvalue: tuple[Component, ...] = ()

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        """Execute the complete native TorchRL module."""

        return Component(self.module)(data)

    def run(self, workflow: Workflow, data: TensorDictBase) -> WorkflowResult:
        """Run a TDHook workflow against the complete native module."""

        return Component(self.module).run(workflow, data)


@singledispatch
def interpret_module(module: TensorDictModuleBase) -> Component | ModuleInterpretation:
    """Interpret known TorchRL module types, preserving generic modules."""

    return Component(module)


@interpret_module.register
def _interpret_probabilistic_actor(module: ProbabilisticActor) -> ModuleInterpretation:
    component = Component(module, "actor")
    return ModuleInterpretation(module=module, actor=component, policy=component)


@interpret_module.register
def _interpret_qvalue_actor(module: QValueActor) -> ModuleInterpretation:
    component = Component(module, "qvalue")
    return ModuleInterpretation(module=module, policy=component, qvalue=(component,))


@interpret_module.register
def _interpret_value_operator(module: ValueOperator) -> ModuleInterpretation:
    if any(_key_leaf(key) == "action" for key in module.in_keys):
        component = Component(module, "qvalue")
        return ModuleInterpretation(module=module, critic=component, qvalue=(component,))
    component = Component(module, "value")
    return ModuleInterpretation(module=module, value=component)


@interpret_module.register
def _interpret_actor_value(module: ActorValueOperator) -> ModuleInterpretation:
    actor = Component(module.get_policy_operator(), "actor")
    value = Component(module.get_value_operator(), "critic")
    return ModuleInterpretation(
        module=module,
        actor=actor,
        policy=actor,
        critic=value,
        value=value,
    )


@interpret_module.register
def _interpret_actor_critic(module: ActorCriticOperator) -> ModuleInterpretation:
    actor = Component(module.get_policy_operator(), "actor")
    critic = Component(module.get_critic_operator(), "critic")
    return ModuleInterpretation(
        module=module,
        actor=actor,
        policy=actor,
        critic=critic,
        qvalue=(critic,),
    )


def _key_leaf(key: NestedKey) -> str:
    return key if isinstance(key, str) else key[-1]


__all__ = ["ModuleInterpretation", "interpret_module"]
