import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.data import Bounded
from torchrl.modules import SafeModule

from xdrl import Interaction


def _interaction(module: torch.nn.Module | None = None) -> Interaction:
    policy = TensorDictModule(
        torch.nn.Linear(2, 1, bias=False) if module is None else module,
        in_keys=["observation"],
        out_keys=["action"],
    )
    return Interaction(policy)


def test_interaction_uses_module_keys_as_the_boundary() -> None:
    interaction = _interaction()
    data = TensorDict({"observation": torch.ones(3, 2)}, batch_size=[3], names=["env"])

    result = interaction(data)

    assert interaction.module.in_keys == ["observation"]
    assert interaction.module.out_keys == ["action"]
    assert result["action"].shape == (3, 1)
    assert result.names == ["env"]


def test_interaction_preserves_torchrl_safe_module_specs() -> None:
    policy = SafeModule(
        lambda observation: observation * 10,
        in_keys=["observation"],
        out_keys=["action"],
        spec=Bounded(low=-1, high=1, shape=(1,)),
        safe=True,
    )
    data = TensorDict({"observation": torch.tensor([[0.5], [-0.5]])}, batch_size=[2])

    result = Interaction(policy)(data)

    torch.testing.assert_close(result["action"], torch.tensor([[1.0], [-1.0]]))


def test_interaction_rejects_invalid_module_and_result_types() -> None:
    with pytest.raises(TypeError, match="TensorDictModuleBase"):
        Interaction(torch.nn.Linear(2, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="interaction input must be a TensorDict"):
        _interaction()(object())  # type: ignore[arg-type]

    class InvalidOutput(TensorDictModuleBase):
        in_keys = ["observation"]
        out_keys = ["action"]

        def forward(self, data: TensorDictBase) -> object:
            return object()

    with pytest.raises(TypeError, match="module must return a TensorDict"):
        Interaction(InvalidOutput())(TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2]))


def test_interaction_preserves_caller_owned_torch_state() -> None:
    class Probe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.states: list[tuple[bool, bool]] = []

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.states.append((self.training, torch.is_grad_enabled()))
            return value[:, :1]

    probe = Probe()
    probe.eval()
    interaction = _interaction(probe)
    data = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])

    with torch.no_grad():
        interaction(data)

    assert probe.states == [(False, False)]
    assert not probe.training
