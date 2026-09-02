import pytest
import torch
from tensordict import TensorDict
from torchrl.data import UnboundedContinuous

from xdrl import BatchSemantics, KeyRole, KeySchema, SchemaValidationError, TensorDictSchema


def test_schema_validates_nested_keys_batch_dimensions_and_specs() -> None:
    schema = TensorDictSchema(
        (KeySchema(("agents", "observation"), KeyRole.OBSERVATION, UnboundedContinuous(shape=(4,))),)
    )
    data = TensorDict(
        {"agents": TensorDict({"observation": torch.zeros(2, 3, 4)}, batch_size=[2, 3])},
        batch_size=[2, 3],
    )

    schema.validate(data, BatchSemantics(("env", "agent")), boundary="input")


def test_schema_reports_missing_and_optional_keys() -> None:
    required = TensorDictSchema((KeySchema(("agents", "action"), KeyRole.ACTION),))
    optional = TensorDictSchema((KeySchema("state", KeyRole.STATE, required=False),))
    data = TensorDict({}, batch_size=[2])

    with pytest.raises(SchemaValidationError, match="agents/action"):
        required.validate(data, BatchSemantics(("env",)))
    optional.validate(data, BatchSemantics(("env",)))


def test_schema_rejects_duplicate_keys_and_batch_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        TensorDictSchema((KeySchema("value", KeyRole.VALUE), KeySchema("value", KeyRole.VALUE)))
    with pytest.raises(ValueError, match="unique"):
        BatchSemantics(("env", "env"))


def test_schema_rejects_feature_shape_and_batch_rank_mismatches() -> None:
    schema = TensorDictSchema((KeySchema("value", KeyRole.VALUE, UnboundedContinuous(shape=(1,))),))

    with pytest.raises(SchemaValidationError, match="batch dimensions"):
        schema.validate(TensorDict({"value": torch.zeros(2, 1)}, batch_size=[2, 1]), BatchSemantics(("env",)))
    with pytest.raises(SchemaValidationError, match="does not satisfy"):
        schema.validate(TensorDict({"value": torch.zeros(2, 2)}, batch_size=[2]), BatchSemantics(("env",)))
