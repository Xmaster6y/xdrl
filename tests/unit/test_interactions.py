from contextlib import contextmanager

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import exploration_type

from xdrl.interactions import (
    InteractionDescriptor,
    InteractionPhase,
    LifecycleEventType,
    RuntimeInteractionContext,
    SchemaSnapshot,
)
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _schemas() -> tuple[TensorDictSchema, TensorDictSchema]:
    return (
        TensorDictSchema(
            (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
        ),
        TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",))),
    )


def _policy() -> TensorDictModule:
    return TensorDictModule(torch.nn.Linear(2, 1, bias=False), in_keys=["observation"], out_keys=["action"])


def _descriptor(
    phase: InteractionPhase, input_schema: TensorDictSchema, output_schema: TensorDictSchema
) -> InteractionDescriptor:
    return InteractionDescriptor(
        identity=f"policy:{phase.value}:7",
        role=ModelRole.ACTOR,
        phase=phase,
        module_path="policy.module",
        input_schema=SchemaSnapshot.from_schema(input_schema),
        output_schema=SchemaSnapshot.from_schema(output_schema),
        batch_dimensions=("env",),
        exploration_mode="random" if phase is InteractionPhase.COLLECTION else "deterministic",
        gradient_enabled=False,
        logical_step=7,
    )


def test_collection_and_evaluation_are_distinct_contexts_for_one_policy() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(3, 2)}, batch_size=[3])
    collection = RuntimeInteractionContext(
        _descriptor(InteractionPhase.COLLECTION, inputs, outputs), _policy(), inputs, outputs, batch
    )
    evaluation = RuntimeInteractionContext(
        _descriptor(InteractionPhase.EVALUATION, inputs, outputs), _policy(), inputs, outputs, batch
    )

    with collection:
        collection.invoke(batch.clone())
    with evaluation:
        evaluation.invoke(batch.clone())

    assert collection.descriptor.identity != evaluation.descriptor.identity
    assert collection.descriptor.exploration_mode != evaluation.descriptor.exploration_mode
    assert [event.kind for event in collection.events] == [LifecycleEventType.BEFORE, LifecycleEventType.AFTER]
    assert [event.order for event in evaluation.events] == [0, 1]


def test_context_restores_execution_and_hook_state_after_exception() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    entered: list[str] = []

    @contextmanager
    def hook_state():
        entered.append("entered")
        try:
            yield
        finally:
            entered.append("exited")

    descriptor = _descriptor(InteractionPhase.COLLECTION, inputs, outputs)
    original_exploration = exploration_type()
    original_grad = torch.is_grad_enabled()
    context = RuntimeInteractionContext(descriptor, _policy(), inputs, outputs, batch, hook_state)
    with pytest.raises(RuntimeError, match="boom"):
        with context:
            assert exploration_type().value == "random"
            assert not torch.is_grad_enabled()
            raise RuntimeError("boom")
    assert exploration_type() == original_exploration
    assert torch.is_grad_enabled() == original_grad
    assert entered == ["entered", "exited"]


class _FailingPolicy:
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        raise RuntimeError("policy failure")


def test_failure_event_retains_only_diagnostics() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(2, 2)}, batch_size=[2])
    context = RuntimeInteractionContext(
        _descriptor(InteractionPhase.EVALUATION, inputs, outputs), _FailingPolicy(), inputs, outputs, batch
    )

    with context, pytest.raises(RuntimeError, match="policy failure"):
        context.invoke(batch)

    failure = context.events[-1]
    assert failure.kind is LifecycleEventType.FAILURE
    assert failure.phase is InteractionPhase.EVALUATION
    assert failure.module_path == "policy.module"
    assert failure.key_shapes == {"observation": (2, 2)}
    assert "policy failure" in failure.error
    assert "tensor" not in repr(failure).lower()
