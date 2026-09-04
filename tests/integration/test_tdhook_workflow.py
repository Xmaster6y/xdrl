import pytest
import torch
from tdhook.execution import ExecutionSpec, GradientMode
from tdhook.latent import ActivationCaching
from tdhook.targets import Target
from tdhook.workflow import Workflow, WorkflowResult
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import Component, interpret


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


def _component() -> Component:
    module = TensorDictModule(ReusedLayer(), in_keys=["observation"], out_keys=["action"])
    return interpret(module)


@pytest.mark.integration
def test_component_run_returns_tdhooks_native_result() -> None:
    component = _component()
    workflow = Workflow(ActivationCaching("module.shared", cache_key=("activations", "all")))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    result = component.run(workflow, data)

    assert isinstance(result, WorkflowResult)
    assert result.plan.model_passes == 1
    assert len(result.data["activations", "all"]["module.shared"]) == 1
    assert not component.module.module.shared._forward_hooks


@pytest.mark.integration
def test_tdhook_owns_repeated_occurrence_selection() -> None:
    component = _component()
    target = Target("module.shared", "activation", -1, (0,), occurrences=(1,))
    workflow = Workflow(ActivationCaching(target, cache_key=("activations", "selected")))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    result = component.run(workflow, data)

    torch.testing.assert_close(result.data["activations", "selected", "module.shared"], torch.tensor([[3.0]]))


@pytest.mark.integration
def test_component_run_validates_module_boundaries() -> None:
    with pytest.raises(ValueError, match="missing TensorDict keys"):
        _component().run(Workflow(ActivationCaching("module.shared")), TensorDict({}, batch_size=[1]))


@pytest.mark.integration
def test_component_run_rejects_gradient_mismatch_before_execution() -> None:
    component = _component()
    workflow = Workflow(GradientCaching("module.shared"))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    with torch.no_grad(), pytest.raises(ValueError, match="requires enabled autograd"):
        component.run(workflow, data)

    assert component.module.module.calls == 0
    assert not component.module.module.shared._forward_hooks


@pytest.mark.integration
def test_component_run_rejects_disabled_gradient_execution_when_enabled() -> None:
    component = _component()
    workflow = Workflow(NoGradientCaching("module.shared"))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    with pytest.raises(ValueError, match="requires a no-grad context"):
        component.run(workflow, data)


def test_component_run_rejects_invalid_workflow() -> None:
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
    with pytest.raises(TypeError, match="workflow must be"):
        _component().run(object(), data)  # type: ignore[arg-type]


def test_component_run_rejects_an_invalid_tdhook_result(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = Workflow(ActivationCaching("module.shared"))
    monkeypatch.setattr(Workflow, "run_with_plan", lambda self, module, data: object())
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    with pytest.raises(TypeError, match="invalid result"):
        _component().run(workflow, data)
