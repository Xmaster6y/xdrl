import json

import torch
from tensordict import TensorDict

from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext, SchemaSnapshot
from xdrl.observations import (
    HookDirection,
    ObservationKind,
    ObservationTrace,
    OverflowPolicy,
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


def _descriptor(inputs: TensorDictSchema, outputs: TensorDictSchema) -> InteractionDescriptor:
    return InteractionDescriptor(
        "trajectory:4:policy",
        ModelRole.ACTOR,
        InteractionPhase.COLLECTION,
        "policy",
        SchemaSnapshot.from_schema(inputs),
        SchemaSnapshot.from_schema(outputs),
        model_id="policy-v2",
        checkpoint_id="checkpoint-4",
        batch_dimensions=("env",),
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
    context = RuntimeInteractionContext(
        _descriptor(inputs, outputs), model, inputs, outputs, batch, observations=trace
    )

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
    trace = ObservationTrace(RetentionPolicy(tensor=TensorRetention.CPU, reduction="mean", max_records=1))
    descriptor = _descriptor(inputs, outputs)
    source = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    first = trace.observe_tensor(
        descriptor, source, kind=ObservationKind.ACTIVATION, target="encoder", direction=HookDirection.OUTPUT
    )
    assert first is not None and first.payload is not None
    assert not first.payload.requires_grad
    assert first.payload.device.type == "cpu"
    assert first.retained_batch_dimensions == ()
    assert torch.equal(first.payload, torch.tensor([3.0, 5.0]))
    trace.observe_tensor(descriptor, source, kind=ObservationKind.GRADIENT, target="encoder")
    assert len(trace.records) == 1
    assert trace.dropped == 1


def test_sampling_streaming_and_backpressure_are_explicit() -> None:
    inputs, outputs = _schemas()
    streamed = []
    trace = ObservationTrace(
        RetentionPolicy(every_n=2, max_records=1, overflow=OverflowPolicy.DROP_NEWEST), callback=streamed.append
    )
    descriptor = _descriptor(inputs, outputs)
    for _ in range(3):
        trace.observe_tensor(descriptor, torch.zeros(1), kind=ObservationKind.TENSORDICT_KEY, target="obs")
    assert len(trace.records) == 1
    assert len(streamed) == 2
    assert trace.dropped == 1
