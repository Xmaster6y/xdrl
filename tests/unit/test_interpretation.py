from typing import ClassVar

import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.data import Bounded
from torchrl.modules import SafeModule

from xdrl import Component, interpret


def _component(module: torch.nn.Module | None = None) -> Component:
    policy = TensorDictModule(
        torch.nn.Linear(2, 1, bias=False) if module is None else module,
        in_keys=["observation"],
        out_keys=["action"],
    )
    return interpret(policy)


def test_interpret_uses_module_keys_as_the_boundary() -> None:
    component = _component()
    data = TensorDict({"observation": torch.ones(3, 2)}, batch_size=[3], names=["env"])

    result = component(data)

    assert component.module.in_keys == ["observation"]
    assert component.module.out_keys == ["action"]
    assert result["action"].shape == (3, 1)
    assert result.names == ["env"]


def test_component_preserves_torchrl_safe_module_specs() -> None:
    policy = SafeModule(
        lambda observation: observation * 10,
        in_keys=["observation"],
        out_keys=["action"],
        spec=Bounded(low=-1, high=1, shape=(1,)),
        safe=True,
    )
    data = TensorDict({"observation": torch.tensor([[0.5], [-0.5]])}, batch_size=[2])

    result = interpret(policy)(data)

    torch.testing.assert_close(result["action"], torch.tensor([[1.0], [-1.0]]))


def test_interpret_rejects_invalid_module_and_result_types() -> None:
    with pytest.raises(TypeError, match="cannot interpret Linear"):
        interpret(torch.nn.Linear(2, 1))
    with pytest.raises(TypeError, match="component input must be a TensorDict"):
        _component()(object())  # type: ignore[arg-type]

    class InvalidOutput(TensorDictModuleBase):
        in_keys: ClassVar = ["observation"]
        out_keys: ClassVar = ["action"]

        def forward(self, data: TensorDictBase) -> object:
            return object()

    with pytest.raises(TypeError, match="module must return a TensorDict"):
        interpret(InvalidOutput())(TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2]))


def test_component_rejects_invalid_identity_and_parameter_source() -> None:
    module = TensorDictModule(torch.nn.Linear(2, 1), ["observation"], ["action"])
    with pytest.raises(TypeError, match="TensorDictModuleBase"):
        Component(torch.nn.Linear(2, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name must be non-empty"):
        Component(module, name="")
    with pytest.raises(TypeError, match="do not support TensorDict.to_module"):
        Component(module, params=object()).parameter_context()  # type: ignore[arg-type]


def test_component_preserves_caller_owned_torch_state() -> None:
    class Probe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.states: list[tuple[bool, bool]] = []

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.states.append((self.training, torch.is_grad_enabled()))
            return value[:, :1]

    probe = Probe()
    probe.eval()
    component = _component(probe)
    data = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])

    with torch.no_grad():
        component(data)

    assert probe.states == [(False, False)]
    assert not probe.training
