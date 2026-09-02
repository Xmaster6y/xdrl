import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import (
    BatchSemantics,
    Interaction,
    InteractionSpec,
    KeyRole,
    KeySchema,
    ModelRole,
    RecurrentSemantics,
    RecurrentStateTransition,
    TensorDictSchema,
)


class RecurrentCell(torch.nn.Module):
    def forward(self, observation: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = state + observation
        return next_state, next_state


def _interaction() -> Interaction:
    inputs = TensorDictSchema(
        (
            KeySchema("observation", KeyRole.OBSERVATION),
            KeySchema("state", KeyRole.STATE),
            KeySchema("reset", KeyRole.TERMINATION),
        )
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION), KeySchema("next_state", KeyRole.STATE)))
    spec = InteractionSpec(
        ModelRole.ACTOR,
        inputs,
        outputs,
        BatchSemantics(("env",)),
        recurrent=RecurrentSemantics(
            (RecurrentStateTransition("state", "next_state"),),
            reset_keys=("reset",),
        ),
    )
    module = TensorDictModule(
        RecurrentCell(),
        in_keys=["observation", "state"],
        out_keys=["action", "next_state"],
    )
    return Interaction(module, spec)


def test_recurrent_transition_is_checked_at_the_boundary() -> None:
    data = TensorDict(
        {
            "observation": torch.ones(2, 3),
            "state": torch.zeros(2, 3),
            "reset": torch.zeros(2, dtype=torch.bool),
        },
        batch_size=[2],
    )

    result = _interaction()(data)

    assert result["next_state"].shape == data["state"].shape


def test_recurrent_reset_must_be_boolean() -> None:
    data = TensorDict(
        {"observation": torch.ones(2, 3), "state": torch.zeros(2, 3), "reset": torch.zeros(2)},
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="boolean"):
        _interaction()(data)


def test_recurrent_keys_must_be_declared_state_boundaries() -> None:
    with pytest.raises(ValueError, match="required state inputs"):
        InteractionSpec(
            ModelRole.ACTOR,
            TensorDictSchema((KeySchema("state", KeyRole.FEATURE),)),
            TensorDictSchema((KeySchema("next_state", KeyRole.STATE),)),
            recurrent=RecurrentSemantics((RecurrentStateTransition("state", "next_state"),)),
        )
