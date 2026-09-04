import pytest
import torch
from tensordict import NonTensorData, TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import LSTMModule, ValueOperator
from torchrl.objectives import LossModule

from xdrl import RecurrentSemantics, RecurrentStateTransition, interpret
from xdrl.modules import ModuleInterpretation


def test_torchrl_lstm_conventions_are_supported_directly() -> None:
    module = LSTMModule(input_size=3, hidden_size=4, in_key="observation", out_key="embedding")
    recurrent = RecurrentSemantics.from_torchrl("recurrent_state_h", "recurrent_state_c")
    component = interpret(module, recurrent=recurrent)
    data = TensorDict(
        {
            "observation": torch.ones(2, 3),
            "recurrent_state_h": torch.zeros(2, 1, 4),
            "recurrent_state_c": torch.zeros(2, 1, 4),
            "is_init": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )

    result = component(data)

    assert result["next", "recurrent_state_h"].shape == data["recurrent_state_h"].shape
    assert result["next", "recurrent_state_c"].shape == data["recurrent_state_c"].shape


def test_recurrent_semantics_preserve_known_module_views() -> None:
    class RecurrentValue(torch.nn.Module):
        def forward(
            self,
            observation: torch.Tensor,
            state: torch.Tensor,
            is_init: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del is_init
            return observation[:, :1], state + 1

    module = ValueOperator(
        RecurrentValue(),
        in_keys=["observation", "state", "is_init"],
        out_keys=["state_value", ("next", "state")],
    )
    recurrent = RecurrentSemantics.from_torchrl("state")
    interpretation = interpret(module, recurrent=recurrent)
    data = TensorDict(
        {
            "observation": torch.ones(2, 3),
            "state": torch.zeros(2, 4),
            "is_init": torch.zeros(2, dtype=torch.bool),
        },
        batch_size=[2],
    )

    assert isinstance(interpretation, ModuleInterpretation)
    assert interpretation.recurrent is recurrent
    assert interpretation.value.recurrent is recurrent
    result = interpretation.value(data)
    torch.testing.assert_close(result["next", "state"], torch.ones(2, 4))


def test_recurrent_semantics_must_match_module_keys() -> None:
    module = TensorDictModule(torch.nn.Identity(), ["state"], ["next_state"])
    with pytest.raises(ValueError, match="is not a module input"):
        interpret(module, recurrent=RecurrentSemantics((RecurrentStateTransition("missing", "next_state"),)))
    with pytest.raises(ValueError, match="is not a module output"):
        interpret(module, recurrent=RecurrentSemantics((RecurrentStateTransition("state", "missing"),)))
    with pytest.raises(ValueError, match="reset key .* is not a module input"):
        interpret(
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
    component = interpret(
        module,
        recurrent=RecurrentSemantics.from_torchrl("state"),
    )
    data = TensorDict(
        {"state": torch.zeros(2, 3), "is_init": torch.zeros(2)},
        batch_size=[2],
    )

    with pytest.raises(ValueError, match="boolean"):
        component(data)


def test_recurrent_transition_rejects_changed_shape() -> None:
    module = TensorDictModule(lambda state: state[:, :-1], ["state"], [("next", "state")])
    component = interpret(
        module,
        recurrent=RecurrentSemantics((RecurrentStateTransition("state", ("next", "state")),)),
    )
    data = TensorDict({"state": torch.zeros(2, 3)}, batch_size=[2])

    with pytest.raises(ValueError, match="changed shape or dtype"):
        component(data)


def test_recurrent_component_can_be_derived_and_requires_tensor_states() -> None:
    module = TensorDictModule(torch.nn.Identity(), ["state"], [("next", "state")])
    recurrent = RecurrentSemantics((RecurrentStateTransition("state", ("next", "state")),))
    component = interpret(module).with_recurrent(recurrent)
    inputs = TensorDict({"state": torch.zeros(1)}, batch_size=[])
    outputs = TensorDict(
        {"state": torch.zeros(1), ("next", "state"): NonTensorData("bad")},
        batch_size=[],
    )

    assert component.recurrent is recurrent
    with pytest.raises(TypeError, match="tensor-valued keys"):
        component.validate_output(inputs, outputs)


def test_recurrent_semantics_must_be_attached_to_a_selected_loss_component() -> None:
    recurrent = RecurrentSemantics((RecurrentStateTransition("state", ("next", "state")),))
    with pytest.raises(TypeError, match="selected loss component"):
        interpret(LossModule(), recurrent=recurrent)


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
