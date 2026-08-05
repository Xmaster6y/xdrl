import json

import pytest
import torch
from tensordict import TensorDict

from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext
from xdrl.observations import (
    DimensionReduction,
    HookDirection,
    ObservationKind,
    ObservationTrace,
    OverflowPolicy,
    ReductionKind,
    RetentionPolicy,
    TensorRetention,
)
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _schemas() -> tuple[TensorDictSchema, TensorDictSchema]:
    return (
        TensorDictSchema(
            (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
        ),
        TensorDictSchema(
            (
                KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),
                KeySchema("value", KeyRole.VALUE, KeyPresence.PRODUCED),
            ),
            BatchSemantics(("env",)),
        ),
    )


class _Policy(torch.nn.Module):
    def forward(self, tensordict: TensorDict) -> TensorDict:
        return tensordict.set("action", tensordict["observation"] + 1).set(
            "value", tensordict["observation"].sum(-1, keepdim=True)
        )


def _contract(inputs: TensorDictSchema, outputs: TensorDictSchema) -> InteractionContract:
    return InteractionContract(
        "trajectory:4:policy",
        ModelRole.ACTOR,
        InteractionPhase.COLLECTION,
        "policy",
        inputs,
        outputs,
        model_id="policy-v2",
        checkpoint_id="checkpoint-4",
        time_dimension="time",
        agent_dimension="agent",
        logical_step=4,
        trajectory_id="trajectory-4",
        exploration_mode="random",
    )


def test_trace_is_serialisable_and_observation_only_preserves_model_output() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
    model = _Policy()
    expected = model(batch.clone())
    trace = ObservationTrace()
    context = RuntimeInteractionContext(_contract(inputs, outputs), model, batch, observations=trace)

    with context:
        actual = context.invoke(batch.clone())

    assert torch.equal(actual["action"], expected["action"])
    assert [record.kind for record in trace.records] == [
        ObservationKind.MODULE_INPUT,
        ObservationKind.ACTION,
        ObservationKind.VALUE,
    ]
    assert trace.records[1].trajectory_id == "trajectory-4"
    assert trace.records[1].model_id == "policy-v2"
    assert trace.records[1].checkpoint_id == "checkpoint-4"
    assert trace.records[1].exploration_mode == "random"
    assert trace.records[1].payload is None
    json.dumps(trace.records[1].to_dict())


def test_retention_detaches_reduces_and_bounds_records() -> None:
    inputs, outputs = _schemas()
    trace = ObservationTrace(
        RetentionPolicy(
            tensor=TensorRetention.CPU,
            reductions=(DimensionReduction("env", ReductionKind.MEAN),),
            max_records=1,
        )
    )
    contract = _contract(inputs, outputs)
    source = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    first = trace.observe_tensor(
        contract,
        source,
        kind=ObservationKind.ACTIVATION,
        target="encoder",
        direction=HookDirection.OUTPUT,
        batch_dimensions=("env",),
    )
    assert first is not None and first.payload is not None
    assert not first.payload.requires_grad
    assert first.payload.device.type == "cpu"
    assert first.retained_batch_dimensions == ()
    assert torch.equal(first.payload, torch.tensor([3.0, 5.0]))
    trace.observe_tensor(contract, source, kind=ObservationKind.GRADIENT, target="encoder", batch_dimensions=("env",))
    assert len(trace.records) == 1
    assert trace.dropped == 1


def test_max_reduction_and_unbatched_hook_tensors_are_handled_explicitly() -> None:
    inputs, outputs = _schemas()
    contract = _contract(inputs, outputs)
    trace = ObservationTrace(
        RetentionPolicy(
            tensor=TensorRetention.DETACHED,
            reductions=(DimensionReduction("env", ReductionKind.MAX),),
        )
    )
    reduced = trace.observe_tensor(
        contract,
        torch.tensor([[1.0, 5.0], [4.0, 3.0]]),
        kind=ObservationKind.ACTIVATION,
        target="encoder",
        batch_dimensions=("env",),
    )
    unbatched = trace.observe_tensor(
        contract, torch.tensor([1.0, 5.0]), kind=ObservationKind.GRADIENT, target="encoder"
    )
    assert reduced is not None and torch.equal(reduced.payload, torch.tensor([4.0, 5.0]))
    assert reduced.retained_batch_dimensions == ()
    assert unbatched is not None and torch.equal(unbatched.payload, torch.tensor([1.0, 5.0]))
    assert unbatched.retained_batch_dimensions == ()


def test_agent_and_time_reductions_are_named_serialised_and_opt_in() -> None:
    policy = RetentionPolicy(
        tensor=TensorRetention.DETACHED,
        reductions=(
            DimensionReduction("time", ReductionKind.MEAN),
            DimensionReduction("agent", ReductionKind.SUM),
        ),
    )
    trace = ObservationTrace(policy)
    inputs, outputs = _schemas()
    record = trace.observe_tensor(
        _contract(inputs, outputs),
        torch.arange(24, dtype=torch.float).reshape(2, 3, 4, 1),
        kind=ObservationKind.ACTIVATION,
        target="shared_encoder",
        batch_dimensions=("env", "time", "agent"),
    )

    assert record is not None and record.payload is not None
    assert record.payload.shape == (2, 1)
    assert torch.equal(record.payload, torch.tensor([[22.0], [70.0]]))
    assert record.retained_batch_dimensions == ("env",)
    assert json.loads(json.dumps(policy.to_dict()))["reductions"] == [
        {"dimension": "time", "kind": "mean"},
        {"dimension": "agent", "kind": "sum"},
    ]


def test_reduction_rejects_an_unnamed_batch_axis() -> None:
    trace = ObservationTrace(
        RetentionPolicy(
            tensor=TensorRetention.DETACHED,
            reductions=(DimensionReduction("agent", ReductionKind.MEAN),),
        )
    )
    inputs, outputs = _schemas()

    with pytest.raises(ValueError, match="agent.*not present"):
        trace.observe_tensor(
            _contract(inputs, outputs),
            torch.zeros(2, 3),
            kind=ObservationKind.ACTIVATION,
            target="encoder",
            batch_dimensions=("env",),
        )


def test_reduction_policy_rejects_invalid_dimensions_and_limits() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DimensionReduction("", ReductionKind.MEAN)
    with pytest.raises(ValueError, match="at most one"):
        RetentionPolicy(
            reductions=(
                DimensionReduction("agent", ReductionKind.MEAN),
                DimensionReduction("agent", ReductionKind.SUM),
            )
        )
    with pytest.raises(ValueError, match="at least 1"):
        RetentionPolicy(every_n=0)
    with pytest.raises(ValueError, match="non-negative"):
        RetentionPolicy(max_records=-1)


def test_composite_schema_parent_captures_nested_tensor_leaves() -> None:
    inputs, outputs = _schemas()
    contract = _contract(inputs, outputs)
    trace = ObservationTrace()
    batch = TensorDict({"agents": TensorDict({"observation": torch.zeros(2, 3)}, batch_size=[2])}, batch_size=[2])

    records = trace.capture_tensordict(
        contract,
        batch,
        direction=HookDirection.INPUT,
        roles={("agents",): KeyRole.OBSERVATION},
    )

    assert len(records) == 1
    assert records[0].key_path == ("agents", "observation")
    assert records[0].kind is ObservationKind.MODULE_INPUT


def test_observation_serialisation_does_not_include_retained_payload() -> None:
    inputs, outputs = _schemas()
    record = ObservationTrace(RetentionPolicy(tensor=TensorRetention.DETACHED)).observe_tensor(
        _contract(inputs, outputs), torch.ones(1), kind=ObservationKind.ACTIVATION, target="encoder"
    )
    assert record is not None and record.payload is not None
    assert "payload" not in record.to_dict()
    json.dumps(record.to_dict())


def test_sampling_streaming_and_backpressure_are_explicit() -> None:
    inputs, outputs = _schemas()
    streamed = []
    trace = ObservationTrace(
        RetentionPolicy(every_n=2, max_records=1, overflow=OverflowPolicy.DROP_NEWEST), callback=streamed.append
    )
    contract = _contract(inputs, outputs)
    for _ in range(3):
        trace.observe_tensor(contract, torch.zeros(1), kind=ObservationKind.TENSORDICT_KEY, target="obs")
    assert len(trace.records) == 1
    assert len(streamed) == 2
    assert trace.dropped == 1
