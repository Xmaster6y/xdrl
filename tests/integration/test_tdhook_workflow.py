from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow

from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext
from xdrl.tdhook import TDHookWorkflowRunner
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


class CountingLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1, bias=False)
        self.calls = 0
        self.training_during_calls: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.training_during_calls.append(self.training)
        return super().forward(value)


class StatefulTensorDictModule(TensorDictModule):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module, in_keys=["observation"], out_keys=["action"])
        self.root_calls = 0

    def forward(self, tensordict: TensorDict, **kwargs: object) -> TensorDict:
        self.root_calls += 1
        return super().forward(tensordict, **kwargs)


def _interaction(*, stateful: bool = False) -> tuple[RuntimeInteractionContext, CountingLinear]:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",)))
    layer = CountingLinear()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0]]))
    policy = (
        StatefulTensorDictModule(layer)
        if stateful
        else TensorDictModule(layer, in_keys=["observation"], out_keys=["action"])
    )
    policy.train()
    batch = TensorDict({"observation": torch.ones(3, 2)}, batch_size=[3])
    contract = InteractionContract(
        "policy:evaluation:workflow",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        inputs,
        outputs,
        model_id="actor-v3",
        checkpoint_id="sha256:workflow",
        module_training=False,
    )
    return RuntimeInteractionContext(contract, policy, batch), layer


@pytest.mark.integration
def test_compatible_capture_methods_share_one_validated_model_call() -> None:
    interaction, layer = _interaction()
    workflow = Workflow(
        ActivationCaching("module", cache_key=("activations", "first")),
        ActivationCaching("module", cache_key=("activations", "second")),
    )
    runner = TDHookWorkflowRunner(interaction)
    plan = runner.plan(workflow, interaction.representative_input.clone())

    execution = runner.run(
        workflow,
        interaction.representative_input.clone(),
        code_revision="test-revision",
        expected_plan=plan,
    )

    assert plan.model_passes == 1
    assert plan.executions[0].coexecuted
    assert layer.calls == 1
    assert layer.training_during_calls == [False]
    assert interaction.module.training
    assert execution.provenance.model_calls == 1
    assert "module" in execution.data.get(("activations", "first"))
    assert "module" in execution.data.get(("activations", "second"))
    assert not layer._forward_hooks


@pytest.mark.integration
def test_workflow_preserves_root_module_type_and_state() -> None:
    interaction, _ = _interaction(stateful=True)
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))

    TDHookWorkflowRunner(interaction).run(
        workflow, interaction.representative_input.clone(), code_revision="test-revision"
    )

    assert type(interaction.module) is StatefulTensorDictModule
    assert interaction.module.root_calls == 1
    assert interaction.module.training


class _ExtraCallWorkflow(Workflow):
    def run_with_plan(self, model: torch.nn.Module, data: TensorDict):  # type: ignore[no-untyped-def]
        result = super().run_with_plan(model, data)
        model(result.data)
        return result


class _MissingPublicEvidenceWorkflow(Workflow):
    @property
    def describe(self):  # type: ignore[no-untyped-def]
        raise AttributeError("describe")


class _InvalidPublicEvidenceWorkflow(Workflow):
    def run_with_plan(self, model: torch.nn.Module, data: TensorDict):  # type: ignore[no-untyped-def]
        return SimpleNamespace(data=object(), plan=super().plan(model, data))


class _DriftingPlanWorkflow(Workflow):
    def run_with_plan(self, model: torch.nn.Module, data: TensorDict):  # type: ignore[no-untyped-def]
        execution = super().run_with_plan(model, data)
        return replace(execution, plan=replace(execution.plan, executions=()))


@pytest.mark.integration
def test_planned_and_observed_model_passes_must_match() -> None:
    interaction, _ = _interaction()
    workflow = _ExtraCallWorkflow(ActivationCaching("module"))

    with pytest.raises(RuntimeError, match="model-pass mismatch"):
        TDHookWorkflowRunner(interaction).run(
            workflow, interaction.representative_input.clone(), code_revision="test-revision"
        )

    assert not interaction.module._forward_hooks


@pytest.mark.integration
def test_runner_consumes_public_execution_plan_and_configured_step_descriptions() -> None:
    interaction, _ = _interaction()
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))

    execution = TDHookWorkflowRunner(interaction).run(
        workflow, interaction.representative_input.clone(), code_revision="test-revision"
    )

    assert execution.plan == workflow.plan(interaction.module, interaction.representative_input.clone())
    assert execution.provenance.configured_steps
    assert "cache_key" in execution.provenance.configured_steps[0]


@pytest.mark.integration
def test_runner_rejects_tdhook_without_public_execution_evidence() -> None:
    interaction, _ = _interaction()
    workflow = Workflow(ActivationCaching("module"))
    workflow.run_with_plan = None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="public execution evidence"):
        TDHookWorkflowRunner(interaction).run(
            workflow, interaction.representative_input.clone(), code_revision="test-revision"
        )


@pytest.mark.integration
def test_runner_rejects_tdhook_with_missing_public_execution_evidence() -> None:
    interaction, _ = _interaction()

    with pytest.raises(RuntimeError, match="public execution evidence"):
        TDHookWorkflowRunner(interaction).run(
            _MissingPublicEvidenceWorkflow(ActivationCaching("module")),
            interaction.representative_input.clone(),
            code_revision="test-revision",
        )


@pytest.mark.integration
def test_runner_rejects_invalid_public_execution_evidence() -> None:
    interaction, _ = _interaction()

    with pytest.raises(TypeError, match="invalid public evidence"):
        TDHookWorkflowRunner(interaction).run(
            _InvalidPublicEvidenceWorkflow(ActivationCaching("module")),
            interaction.representative_input.clone(),
            code_revision="test-revision",
        )


@pytest.mark.integration
def test_runner_rejects_public_execution_plan_drift() -> None:
    interaction, _ = _interaction()

    with pytest.raises(RuntimeError, match="plan changed during execution"):
        TDHookWorkflowRunner(interaction).run(
            _DriftingPlanWorkflow(ActivationCaching("module")),
            interaction.representative_input.clone(),
            code_revision="test-revision",
        )


@pytest.mark.integration
def test_runner_rejects_an_operator_that_reuses_the_interaction_module_before_execution() -> None:
    interaction, layer = _interaction()
    data = interaction.representative_input.clone()
    workflow = Workflow(interaction.module)

    with pytest.raises(ValueError, match="operators must not invoke the interaction module"):
        TDHookWorkflowRunner(interaction).run(workflow, data, code_revision="test-revision")

    assert layer.calls == 0
    assert not interaction.events
