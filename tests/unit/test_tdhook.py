import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.execution import ExecutionSpec, GradientMode
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow

from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext, SchemaSnapshot
from xdrl.tdhook import TDHookWorkflowRunner
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _interaction(*, lazy: bool = False) -> RuntimeInteractionContext:
    inputs = TensorDictSchema(
        (KeySchema(("agents", "observation"), KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
        BatchSemantics(("env", "agent")),
    )
    outputs = TensorDictSchema(
        (KeySchema(("agents", "action"), KeyRole.ACTION, KeyPresence.PRODUCED),),
        BatchSemantics(("env", "agent")),
    )
    layer: torch.nn.Module = torch.nn.LazyLinear(1, bias=False) if lazy else torch.nn.Linear(2, 1, bias=False)
    policy = TensorDictModule(layer, in_keys=[("agents", "observation")], out_keys=[("agents", "action")])
    batch = TensorDict({("agents", "observation"): torch.ones(3, 2, 2)}, batch_size=[3, 2])
    descriptor = InteractionDescriptor(
        "policy:evaluation:0",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        SchemaSnapshot.from_schema(inputs),
        SchemaSnapshot.from_schema(outputs),
        batch_dimensions=("env", "agent"),
    )
    return RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)


def test_workflow_preserves_nested_batched_tensordict_execution() -> None:
    interaction = _interaction()
    with torch.no_grad():
        interaction.module.module.weight.copy_(torch.tensor([[2.0, -1.0]]))
    data = interaction.representative_input.clone()
    expected = interaction.module(data.clone()).get(("agents", "action")).clone()
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))

    execution = TDHookWorkflowRunner(interaction).run(workflow, data)

    assert torch.equal(execution.data.get(("agents", "action")), expected)
    assert "module" in execution.data.get(("activations", "head"))
    assert execution.plan.model_passes == 1
    assert execution.record.model_calls == 1
    assert not interaction.module.module._forward_hooks


def test_runner_rejects_contract_mismatches_and_invalid_boundaries() -> None:
    runner = TDHookWorkflowRunner(_interaction())
    with pytest.raises(TypeError, match="workflow must"):
        runner.run(object(), runner.interaction.representative_input)  # type: ignore[arg-type]
    bad = TensorDict({"other": torch.ones(3, 2, 2)}, batch_size=[3, 2])
    with pytest.raises(ValueError, match="missing required key"):
        runner.plan(Workflow(ActivationCaching("module")), bad)


def test_lazy_modules_require_explicit_materialisation() -> None:
    runner = TDHookWorkflowRunner(_interaction(lazy=True))
    workflow = Workflow(ActivationCaching("module"))
    with pytest.raises(RuntimeError, match="materialize"):
        runner.plan(workflow, runner.interaction.representative_input)
    runner.materialize()
    runner.run(workflow, runner.interaction.representative_input.clone())


def test_runner_rejects_compiled_descendants() -> None:
    interaction = _interaction()
    interaction.module.module._orig_mod = interaction.module.module
    with pytest.raises(NotImplementedError, match="torch.compile"):
        TDHookWorkflowRunner(interaction)


class _RequiredGradientCaching(ActivationCaching):
    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED)


def test_workflow_gradient_requirements_must_match_the_rl_interaction() -> None:
    runner = TDHookWorkflowRunner(_interaction())
    workflow = Workflow(_RequiredGradientCaching("module"))
    with pytest.raises(ValueError, match="gradient-required"):
        runner.run(workflow, runner.interaction.representative_input.clone())
