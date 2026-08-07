from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.execution import AutogradLifetime, ExecutionSpec, GradientMode
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow, WorkflowPlan

from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext
from xdrl.provenance import ProvenanceSchemaError
from xdrl.tdhook import TDHookWorkflowRunner, _configured_step_description
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
    contract = InteractionContract(
        "policy:evaluation:0",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        inputs,
        outputs,
    )
    return RuntimeInteractionContext(contract, policy, batch)


def test_workflow_preserves_nested_batched_tensordict_execution() -> None:
    interaction = _interaction()
    with torch.no_grad():
        interaction.module.module.weight.copy_(torch.tensor([[2.0, -1.0]]))
    data = interaction.representative_input.clone()
    expected = interaction.module(data.clone()).get(("agents", "action")).clone()
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))

    execution = TDHookWorkflowRunner(interaction).run(workflow, data, code_revision="test-revision")

    assert torch.equal(execution.data.get(("agents", "action")), expected)
    assert "module" in execution.data.get(("activations", "head"))
    assert execution.plan.model_passes == 1
    assert execution.provenance.model_calls == 1
    assert not interaction.module.module._forward_hooks


def test_runner_rejects_contract_mismatches_and_invalid_boundaries() -> None:
    runner = TDHookWorkflowRunner(_interaction())
    with pytest.raises(TypeError, match="workflow must"):
        runner.run(object(), runner.interaction.representative_input, code_revision="test-revision")  # type: ignore[arg-type]
    bad = TensorDict({"other": torch.ones(3, 2, 2)}, batch_size=[3, 2])
    with pytest.raises(ValueError, match="missing required key"):
        runner.plan(Workflow(ActivationCaching("module")), bad)
    with pytest.raises(TypeError, match="workflow data must"):
        runner.plan(Workflow(ActivationCaching("module")), object())  # type: ignore[arg-type]


def test_runner_rejects_invalid_provenance_metadata_before_execution() -> None:
    runner = TDHookWorkflowRunner(_interaction())
    workflow = Workflow(ActivationCaching("module"))

    with pytest.raises(ProvenanceSchemaError, match="code_revision"):
        runner.run(workflow, runner.interaction.representative_input.clone(), code_revision=" ")

    assert not runner.interaction.events
    assert not runner.interaction.module.module._forward_hooks


def test_lazy_modules_require_explicit_materialisation() -> None:
    runner = TDHookWorkflowRunner(_interaction(lazy=True))
    workflow = Workflow(ActivationCaching("module"))
    with pytest.raises(RuntimeError, match="materialize"):
        runner.plan(workflow, runner.interaction.representative_input)
    runner.materialize()
    runner.run(workflow, runner.interaction.representative_input.clone(), code_revision="test-revision")


def test_runner_rejects_compiled_descendants() -> None:
    interaction = _interaction()
    interaction.module.module._orig_mod = interaction.module.module
    with pytest.raises(NotImplementedError, match="torch.compile"):
        TDHookWorkflowRunner(interaction)


def test_configured_step_descriptions_must_be_serializable_public_data() -> None:
    with pytest.raises(TypeError, match="expose to_dict"):
        _configured_step_description(object())

    class _NonSerializableDescription:
        def to_dict(self) -> object:
            return {"callback": object()}

    with pytest.raises(TypeError, match="must be JSON-compatible"):
        _configured_step_description(_NonSerializableDescription())


class _RequiredGradientCaching(ActivationCaching):
    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED)


def test_workflow_gradient_requirements_must_match_the_rl_interaction() -> None:
    runner = TDHookWorkflowRunner(_interaction())
    workflow = Workflow(_RequiredGradientCaching("module"))
    with pytest.raises(ValueError, match="gradient-required"):
        runner.run(workflow, runner.interaction.representative_input.clone(), code_revision="test-revision")


def test_gradient_required_execution_without_a_lifetime_declaration_is_rejected() -> None:
    runner = TDHookWorkflowRunner(_interaction())
    plan = SimpleNamespace(executions=(SimpleNamespace(gradient_mode=GradientMode.REQUIRED),))

    with pytest.raises(ValueError, match="autograd lifetime declaration"):
        runner._validate_gradient_contract(plan)  # type: ignore[arg-type]


class _DeferredBackwardCaching(ActivationCaching):
    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED, autograd_lifetime=AutogradLifetime.BACKWARD)


class _InternalBackwardWorkflow(Workflow):
    def run_with_plan(self, model: torch.nn.Module, data: TensorDict):  # type: ignore[no-untyped-def]
        result = super().run_with_plan(model, data)
        result.data.get(("agents", "action")).sum().backward()
        return result


def test_gradient_required_call_lifetime_workflow_runs_when_xdrl_owns_enabled_gradients() -> None:
    interaction = _interaction()
    interaction.contract = replace(interaction.contract, gradient_enabled=True)
    runner = TDHookWorkflowRunner(interaction)

    execution = runner.run(
        _InternalBackwardWorkflow(_RequiredGradientCaching("module")),
        interaction.representative_input.clone(),
        code_revision="test-revision",
    )

    assert execution.plan.executions[0].autograd_lifetime is AutogradLifetime.CALL
    assert interaction.module.module.weight.grad is not None
    assert not interaction.module.module._forward_hooks


def test_gradient_required_call_lifetime_workflow_requires_enabled_gradients() -> None:
    runner = TDHookWorkflowRunner(_interaction())

    with pytest.raises(ValueError, match="requires gradient_enabled=True"):
        runner.run(
            _InternalBackwardWorkflow(_RequiredGradientCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
        )

    assert not runner.interaction.events


def test_gradient_required_call_lifetime_workflow_rejects_inference_mode() -> None:
    interaction = _interaction()
    interaction.contract = replace(interaction.contract, inference_mode=True)
    runner = TDHookWorkflowRunner(interaction)

    with pytest.raises(ValueError, match="incompatible with inference_mode=True"):
        runner.run(
            _InternalBackwardWorkflow(_RequiredGradientCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
        )

    assert not runner.interaction.events


def test_deferred_backward_workflow_remains_rejected() -> None:
    interaction = _interaction()
    interaction.contract = replace(interaction.contract, gradient_enabled=True)
    runner = TDHookWorkflowRunner(interaction)

    with pytest.raises(ValueError, match="deferred-backward"):
        runner.run(
            Workflow(_DeferredBackwardCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
        )

    assert not runner.interaction.events


class _DisabledGradientCaching(ActivationCaching):
    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(gradient_mode=GradientMode.DISABLED)


def test_gradient_disabled_workflow_rejects_gradient_enabled_interaction() -> None:
    interaction = _interaction()
    interaction.contract = replace(interaction.contract, gradient_enabled=True)
    runner = TDHookWorkflowRunner(interaction)

    with pytest.raises(ValueError, match="gradient-disabled"):
        runner.run(
            Workflow(_DisabledGradientCaching("module")),
            interaction.representative_input.clone(),
            code_revision="test-revision",
        )


def test_runner_rejects_plan_drift_before_execution() -> None:
    runner = TDHookWorkflowRunner(_interaction())

    with pytest.raises(RuntimeError, match="plan changed"):
        runner.run(
            Workflow(ActivationCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
            expected_plan=WorkflowPlan((), ()),
        )

    assert not runner.interaction.events


def test_runner_rejects_data_parallel_modules() -> None:
    interaction = _interaction()
    interaction.module = torch.nn.DataParallel(interaction.module)

    with pytest.raises(NotImplementedError, match="distributed modules"):
        TDHookWorkflowRunner(interaction)
