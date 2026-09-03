import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import Interaction


def test_interaction_preserves_native_tensordict_execution() -> None:
    module = TensorDictModule(torch.nn.Linear(3, 2), in_keys=["observation"], out_keys=["action"])
    data = TensorDict({"observation": torch.randn(4, 3)}, batch_size=[4])
    expected = module(data.clone())

    actual = Interaction(module)(data.clone())

    expected_keys = set(expected.keys(include_nested=True, leaves_only=True))
    actual_keys = set(actual.keys(include_nested=True, leaves_only=True))
    assert actual_keys == expected_keys
    for key in expected_keys:
        torch.testing.assert_close(actual[key], expected[key])
        assert actual[key].requires_grad is expected[key].requires_grad
