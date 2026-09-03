import pytest
import torch
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import Interaction, run_workflow


@pytest.mark.integration
def test_quickstart_captures_hidden_activation() -> None:
    policy = TensorDictModule(
        torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        ),
        in_keys=["observation"],
        out_keys=["action"],
    )
    batch = TensorDict(
        {"observation": torch.randn(8, 4)},
        batch_size=[8],
    )
    interaction = Interaction(policy)
    workflow = Workflow(ActivationCaching("module.1", cache_key=("activations", "hidden")))

    execution = run_workflow(interaction, workflow, batch)

    assert execution.data["action"].shape == (8, 2)
    assert execution.data["activations", "hidden", "module.1"].shape == (8, 8)
