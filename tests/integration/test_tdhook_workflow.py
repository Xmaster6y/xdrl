import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.targets import Target
from tdhook.workflow import Workflow, WorkflowResult

from xdrl import (
    BatchSemantics,
    Interaction,
    InteractionSpec,
    KeyRole,
    KeySchema,
    ModelRole,
    SchemaValidationError,
    TensorDictSchema,
    run_workflow,
)


class ReusedLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.shared(value + 1) + self.shared(value + 2)


def _interaction() -> Interaction:
    module = TensorDictModule(ReusedLayer(), in_keys=["observation"], out_keys=["action"])
    spec = InteractionSpec(
        ModelRole.ACTOR,
        TensorDictSchema((KeySchema("observation", KeyRole.OBSERVATION),)),
        TensorDictSchema((KeySchema("action", KeyRole.ACTION),)),
        BatchSemantics(("env",)),
        training=False,
    )
    return Interaction(module, spec)


@pytest.mark.integration
def test_run_workflow_returns_tdhooks_native_result() -> None:
    interaction = _interaction()
    workflow = Workflow(ActivationCaching("module.shared", cache_key=("activations", "all")))
    data = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

    result = run_workflow(interaction, workflow, data)

    assert isinstance(result, WorkflowResult)
    assert result.plan.model_passes == 1
    assert len(result.data["activations", "all"]["module.shared"]) == 1
    assert interaction.module.training
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
def test_run_workflow_validates_xdrl_boundaries() -> None:
    with pytest.raises(SchemaValidationError, match="interaction input"):
        run_workflow(_interaction(), Workflow(ActivationCaching("module.shared")), TensorDict({}, batch_size=[1]))
