import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.execution import ExecutionSpec, GradientMode
from tdhook.latent import ActivationCaching
from tdhook.targets import Target
from tdhook.workflow import Workflow, WorkflowResult

from xdrl import Interaction, run_workflow


class ReusedLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Identity()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.shared(value + 1) + self.shared(value + 2)


class GradientCaching(ActivationCaching):
    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED)


class NoGradientCaching(ActivationCaching):
    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(gradient_mode=GradientMode.DISABLED)


def _interaction() -> Interaction:
    module = TensorDictModule(ReusedLayer(), in_keys=["observation"], out_keys=["action"])
    return Interaction(module)


@pytest.mark.integration
def test_run_workflow_returns_tdhooks_native_result() -> None:
    interaction = _interaction()
    workflow = Workflow(ActivationCaching("module.shared", cache_key=("activations", "all")))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    result = run_workflow(interaction, workflow, data)

    assert isinstance(result, WorkflowResult)
    assert result.plan.model_passes == 1
    assert len(result.data["activations", "all"]["module.shared"]) == 1
    assert not interaction.module.module.shared._forward_hooks


@pytest.mark.integration
def test_tdhook_owns_repeated_occurrence_selection() -> None:
    interaction = _interaction()
    target = Target("module.shared", "activation", -1, (0,), occurrence=1)
    workflow = Workflow(ActivationCaching(target, cache_key=("activations", "selected")))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    result = run_workflow(interaction, workflow, data)

    torch.testing.assert_close(result.data["activations", "selected", "module.shared"], torch.tensor([[3.0]]))


@pytest.mark.integration
def test_run_workflow_validates_module_boundaries() -> None:
    with pytest.raises(ValueError, match="missing TensorDict keys"):
        run_workflow(_interaction(), Workflow(ActivationCaching("module.shared")), TensorDict({}, batch_size=[1]))


@pytest.mark.integration
def test_run_workflow_rejects_gradient_mismatch_before_execution() -> None:
    interaction = _interaction()
    workflow = Workflow(GradientCaching("module.shared"))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    with torch.no_grad(), pytest.raises(ValueError, match="requires enabled autograd"):
        run_workflow(interaction, workflow, data)

    assert interaction.module.module.calls == 0
    assert not interaction.module.module.shared._forward_hooks


@pytest.mark.integration
def test_run_workflow_rejects_disabled_gradient_execution_when_enabled() -> None:
    interaction = _interaction()
    workflow = Workflow(NoGradientCaching("module.shared"))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    with pytest.raises(ValueError, match="requires a no-grad context"):
        run_workflow(interaction, workflow, data)


def test_run_workflow_rejects_invalid_entrypoint_arguments() -> None:
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
    with pytest.raises(TypeError, match="interaction must be"):
        run_workflow(object(), Workflow(ActivationCaching("module.shared")), data)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="workflow must be"):
        run_workflow(_interaction(), object(), data)  # type: ignore[arg-type]
