import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import BatchSemantics, Interaction, InteractionSpec, KeyRole, KeySchema, ModelRole, TensorDictSchema


def test_typed_interaction_preserves_native_output() -> None:
    layer = torch.nn.Linear(3, 2)
    module = TensorDictModule(layer, in_keys=["observation"], out_keys=["action"])
    data = TensorDict({"observation": torch.randn(4, 3)}, batch_size=[4])
    expected = module(data.clone())
    interaction = Interaction(
        module,
        InteractionSpec(
            ModelRole.ACTOR,
            TensorDictSchema((KeySchema("observation", KeyRole.OBSERVATION),)),
            TensorDictSchema((KeySchema("action", KeyRole.ACTION),)),
            BatchSemantics(("env",)),
            training=module.training,
            gradient_enabled=torch.is_grad_enabled(),
        ),
    )

    actual = interaction(data.clone())

    torch.testing.assert_close(actual["action"], expected["action"])
