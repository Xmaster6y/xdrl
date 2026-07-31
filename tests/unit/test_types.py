import pytest
import torch
from tensordict import TensorDict
from torchrl.data import UnboundedContinuous

from xdrl.types import (
    BatchSemantics,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    SchemaValidationError,
    TensorDictSchema,
    validate_module,
)


def test_nested_key_and_vectorised_batch_are_validated_separately() -> None:
    schema = TensorDictSchema(
        keys=(
            KeySchema(
                ("agents", "observation"), KeyRole.OBSERVATION, KeyPresence.REQUIRED, UnboundedContinuous(shape=(4,))
            ),
        ),
        batch=BatchSemantics(("env", "agent")),
    )
    batch = TensorDict(
        {"agents": TensorDict({"observation": torch.zeros(2, 3, 4)}, batch_size=[2, 3])}, batch_size=[2, 3]
    )

    schema.validate_inputs(batch)


def test_validation_reports_missing_nested_path() -> None:
    schema = TensorDictSchema(
        keys=(KeySchema(("agents", "action"), KeyRole.ACTION, KeyPresence.REQUIRED),),
        batch=BatchSemantics(("env",)),
    )
    with pytest.raises(SchemaValidationError, match="agents/action"):
        schema.validate_inputs(TensorDict({}, batch_size=[2]))


def test_validation_reports_spec_and_batch_mismatches() -> None:
    schema = TensorDictSchema(
        keys=(KeySchema("value", KeyRole.VALUE, KeyPresence.PRODUCED, UnboundedContinuous(shape=(1,))),),
        batch=BatchSemantics(("env",)),
    )
    with pytest.raises(SchemaValidationError, match="batch dimensions mismatch"):
        schema.validate_outputs(TensorDict({"value": torch.zeros(2, 1)}, batch_size=[2, 1]))
    with pytest.raises(SchemaValidationError, match="spec mismatch at value"):
        schema.validate_outputs(TensorDict({"value": torch.zeros(2, 2)}, batch_size=[2]))


class _ValueModule:
    role = ModelRole.VALUE
    input_schema = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    output_schema = TensorDictSchema(
        (KeySchema("value", KeyRole.VALUE, KeyPresence.PRODUCED),), BatchSemantics(("env",))
    )

    def __call__(self, tensordict: TensorDict, *args: object, **kwargs: object) -> TensorDict:
        return tensordict.set("value", torch.zeros(*tensordict.batch_size, 1))


def test_contract_module_validates_composition() -> None:
    module = _ValueModule()
    result = validate_module(module, TensorDict({"observation": torch.zeros(2, 4)}, batch_size=[2]))
    assert result.get("value").shape == (2, 1)
