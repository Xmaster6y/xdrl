"""Semantic views inferred from native TorchRL module classes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

from tdhook.workflow import Workflow, WorkflowResult
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.modules import ActorCriticOperator, ActorValueOperator, ProbabilisticActor, QValueActor, ValueOperator

from xdrl.interpretation import Component, RecurrentSemantics


@dataclass(frozen=True, slots=True)
class ModuleInterpretation:
    """RL functions already expressed by one TorchRL module."""

    module: TensorDictModuleBase
    actor: Component | None = None
    policy: Component | None = None
    critic: Component | None = None
    value: Component | None = None
    qvalue: tuple[Component, ...] = ()
    recurrent: RecurrentSemantics | None = None

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        """Execute the complete native TorchRL module."""

        return Component(self.module, recurrent=self.recurrent)(data)

    def run(self, workflow: Workflow, data: TensorDictBase) -> WorkflowResult:
        """Run a TDHook workflow against the complete native module."""

        return Component(self.module, recurrent=self.recurrent).run(workflow, data)


@singledispatch
def interpret_module(
    module: TensorDictModuleBase,
    recurrent: RecurrentSemantics | None = None,
) -> Component | ModuleInterpretation:
    """Interpret known TorchRL module types, preserving generic modules."""

    return Component(module, recurrent=recurrent)


@interpret_module.register
def _interpret_probabilistic_actor(
    module: ProbabilisticActor,
    recurrent: RecurrentSemantics | None = None,
) -> ModuleInterpretation:
    component = Component(module, "actor", recurrent=recurrent)
    return ModuleInterpretation(module=module, actor=component, policy=component, recurrent=recurrent)


@interpret_module.register
def _interpret_qvalue_actor(
    module: QValueActor,
    recurrent: RecurrentSemantics | None = None,
) -> ModuleInterpretation:
    component = Component(module, "qvalue", recurrent=recurrent)
    return ModuleInterpretation(module=module, policy=component, qvalue=(component,), recurrent=recurrent)


@interpret_module.register
def _interpret_value_operator(
    module: ValueOperator,
    recurrent: RecurrentSemantics | None = None,
) -> ModuleInterpretation:
    if any(_key_leaf(key) == "action" for key in module.in_keys):
        component = Component(module, "qvalue", recurrent=recurrent)
        return ModuleInterpretation(
            module=module,
            critic=component,
            qvalue=(component,),
            recurrent=recurrent,
        )
    component = Component(module, "value", recurrent=recurrent)
    return ModuleInterpretation(module=module, value=component, recurrent=recurrent)


@interpret_module.register
def _interpret_actor_value(
    module: ActorValueOperator,
    recurrent: RecurrentSemantics | None = None,
) -> ModuleInterpretation:
    actor = Component(module.get_policy_operator(), "actor", recurrent=recurrent)
    value = Component(module.get_value_operator(), "critic", recurrent=recurrent)
    return ModuleInterpretation(
        module=module,
        actor=actor,
        policy=actor,
        critic=value,
        value=value,
        recurrent=recurrent,
    )


@interpret_module.register
def _interpret_actor_critic(
    module: ActorCriticOperator,
    recurrent: RecurrentSemantics | None = None,
) -> ModuleInterpretation:
    actor = Component(module.get_policy_operator(), "actor", recurrent=recurrent)
    critic = Component(module.get_critic_operator(), "critic", recurrent=recurrent)
    return ModuleInterpretation(
        module=module,
        actor=actor,
        policy=actor,
        critic=critic,
        qvalue=(critic,),
        recurrent=recurrent,
    )


def _key_leaf(key: NestedKey) -> str:
    return key if isinstance(key, str) else key[-1]


__all__ = ["ModuleInterpretation", "interpret_module"]
