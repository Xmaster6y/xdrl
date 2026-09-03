import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import LSTMModule

from xdrl import Interaction, RecurrentSemantics, RecurrentStateTransition


def test_torchrl_lstm_conventions_are_supported_directly() -> None:
    module = LSTMModule(input_size=3, hidden_size=4, in_key="observation", out_key="embedding")
    recurrent = RecurrentSemantics.from_torchrl("recurrent_state_h", "recurrent_state_c")
    interaction = Interaction(module, recurrent)
    data = TensorDict(
        {
            "observation": torch.ones(2, 3),
            "recurrent_state_h": torch.zeros(2, 1, 4),
            "recurrent_state_c": torch.zeros(2, 1, 4),
            "is_init": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )

    result = interaction(data)

    assert result["next", "recurrent_state_h"].shape == data["recurrent_state_h"].shape
    assert result["next", "recurrent_state_c"].shape == data["recurrent_state_c"].shape


def test_recurrent_semantics_must_match_module_keys() -> None:
    module = TensorDictModule(torch.nn.Identity(), ["state"], ["next_state"])
    with pytest.raises(ValueError, match="is not a module input"):
        Interaction(module, recurrent=RecurrentSemantics((RecurrentStateTransition("missing", "next_state"),)))
    with pytest.raises(ValueError, match="is not a module output"):
        Interaction(module, recurrent=RecurrentSemantics((RecurrentStateTransition("state", "missing"),)))
    with pytest.raises(ValueError, match="reset key .* is not a module input"):
        Interaction(
            module,
            recurrent=RecurrentSemantics(
                (RecurrentStateTransition("state", "next_state"),),
                reset_keys=("is_init",),
            ),
        )


def test_recurrent_reset_must_be_boolean() -> None:
    module = TensorDictModule(
        lambda state: state,
        ["state", "is_init"],
        [("next", "state")],
    )
    interaction = Interaction(
        module,
        RecurrentSemantics.from_torchrl("state"),
    )
    data = TensorDict(
        {"state": torch.zeros(2, 3), "is_init": torch.zeros(2)},
        batch_size=[2],
    )

    with pytest.raises(ValueError, match="boolean"):
        interaction(data)


def test_recurrent_transition_rejects_changed_shape() -> None:
    module = TensorDictModule(lambda state: state[:, :-1], ["state"], [("next", "state")])
    interaction = Interaction(
        module,
        RecurrentSemantics((RecurrentStateTransition("state", ("next", "state")),)),
    )
    data = TensorDict({"state": torch.zeros(2, 3)}, batch_size=[2])

    with pytest.raises(ValueError, match="changed shape or dtype"):
        interaction(data)


def test_recurrent_semantics_reject_empty_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RecurrentSemantics(())
    with pytest.raises(ValueError, match="transition keys must be unique"):
        RecurrentSemantics(
            (
                RecurrentStateTransition("state", "next_state"),
                RecurrentStateTransition("state", "other_next_state"),
            )
        )
    with pytest.raises(ValueError, match="reset keys must be unique"):
        RecurrentSemantics(
            (RecurrentStateTransition("state", "next_state"),),
            reset_keys=("is_init", "is_init"),
        )
