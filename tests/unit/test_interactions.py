from contextlib import contextmanager
from dataclasses import replace
import json

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import exploration_type
from torchrl.data import Bounded

from xdrl.interactions import (
    InteractionDescriptor,
    InteractionPhase,
    LifecycleEventType,
    RuntimeInteractionContext,
    SchemaSnapshot,
)
from xdrl.interventions import (
    Intervention,
    InterventionController,
    InterventionTarget,
    InterventionTiming,
    InterventionValidationError,
    run_paired,
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


def test_context_enables_gradients_inside_outer_inference_mode() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    descriptor = _descriptor(InteractionPhase.OPTIMISATION, inputs, outputs)
    descriptor = replace(descriptor, gradient_enabled=True, inference_mode=False)
    context = RuntimeInteractionContext(descriptor, _policy(), inputs, outputs, batch)

    with torch.inference_mode():
        with context:
            assert torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
        assert torch.is_inference_mode_enabled()


def test_schema_snapshot_preserves_spec_constraints() -> None:
    lower = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED, Bounded(low=-1, high=1, shape=(2,))),),
        BatchSemantics(("env",)),
    )
    wider = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED, Bounded(low=-2, high=2, shape=(2,))),),
        BatchSemantics(("env",)),
    )

    lower_key = SchemaSnapshot.from_schema(lower).keys[0]
    wider_key = SchemaSnapshot.from_schema(wider).keys[0]
    assert lower_key.spec_type == wider_key.spec_type
    assert lower_key.feature_shape == wider_key.feature_shape
    assert lower_key.spec_constraints != wider_key.spec_constraints
    descriptor = InteractionDescriptor(
        "policy:output:0",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy.module",
        SchemaSnapshot.from_schema(lower),
        SchemaSnapshot.from_schema(wider),
    )
    json.dumps(descriptor.to_dict())


def test_context_rejects_schema_that_disagrees_with_descriptor() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    different_outputs = TensorDictSchema(
        (KeySchema("different_action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",))
    )

    with pytest.raises(ValueError, match="output_schema does not match"):
        RuntimeInteractionContext(
            _descriptor(InteractionPhase.EVALUATION, inputs, outputs), _policy(), inputs, different_outputs, batch
        )


def test_descriptor_model_identity_does_not_shift_existing_positional_fields() -> None:
    inputs, outputs = _schemas()
    descriptor = InteractionDescriptor(
        "policy:collection:1",
        ModelRole.ACTOR,
        InteractionPhase.COLLECTION,
        "policy.module",
        SchemaSnapshot.from_schema(inputs),
        SchemaSnapshot.from_schema(outputs),
        ("env",),
        "CartPole",
    )

    assert descriptor.batch_dimensions == ("env",)
    assert descriptor.environment == "CartPole"
    assert descriptor.model_id is None


class _FailingPolicy:
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        raise RuntimeError("policy failure")


class _InvalidOutputPolicy:
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return TensorDict({"unexpected": torch.zeros(*tensordict.batch_size, 3)}, batch_size=tensordict.batch_size)


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


def test_output_validation_failure_reports_output_shapes() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(2, 2)}, batch_size=[2])
    context = RuntimeInteractionContext(
        _descriptor(InteractionPhase.EVALUATION, inputs, outputs), _InvalidOutputPolicy(), inputs, outputs, batch
    )

    with context, pytest.raises(Exception, match="missing produced key"):
        context.invoke(batch)

    assert context.events[-1].kind is LifecycleEventType.FAILURE
    assert context.events[-1].key_shapes == {"unexpected": (2, 3)}


def test_tensordict_interventions_are_checked_and_record_checkpoint_provenance() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
    descriptor = replace(_descriptor(InteractionPhase.EVALUATION, inputs, outputs), checkpoint_id="checkpoint-9")
    no_op = Intervention(
        "control",
        InterventionTarget.TENSORDICT,
        InterventionTiming.OUTPUT,
        transform=lambda value: value.clone(),
        key="action",
    )
    changed = Intervention(
        "steer",
        InterventionTarget.TENSORDICT,
        InterventionTiming.OUTPUT,
        transform=lambda value: value + 1,
        key="action",
    )
    policy = _policy()
    with torch.no_grad():
        policy.module.weight.zero_()
    baseline = RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)
    control = RuntimeInteractionContext(
        descriptor, policy, inputs, outputs, batch, interventions=InterventionController((no_op,))
    )
    steered = RuntimeInteractionContext(
        descriptor, policy, inputs, outputs, batch, interventions=InterventionController((changed,))
    )

    control_result = run_paired(baseline, control, batch)
    result = run_paired(baseline, steered, batch)
    expected = control_result.baseline.get("action")
    assert torch.equal(control_result.intervention.get("action"), expected)
    actual = result.intervention.get("action")

    assert torch.equal(actual, expected + 1)
    assert (result.interaction_id, result.checkpoint_id) == (descriptor.identity, "checkpoint-9")
    record = steered.interventions.records[0]
    assert (record.interaction_id, record.checkpoint_id) == (descriptor.identity, "checkpoint-9")


def test_invalid_tensordict_interventions_fail_at_the_contract_boundary() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
    invalid = Intervention(
        "bad-shape",
        InterventionTarget.TENSORDICT,
        InterventionTiming.INPUT,
        transform=lambda value: value[..., :1],
        key="observation",
    )
    context = RuntimeInteractionContext(
        _descriptor(InteractionPhase.EVALUATION, inputs, outputs),
        _policy(),
        inputs,
        outputs,
        batch,
        interventions=InterventionController((invalid,)),
    )

    with context, pytest.raises(InterventionValidationError, match="must preserve shape"):
        context.invoke(batch.clone())

    with pytest.raises(InterventionValidationError, match="declared required key"):
        RuntimeInteractionContext(
            _descriptor(InteractionPhase.EVALUATION, inputs, outputs),
            _policy(),
            inputs,
            outputs,
            batch,
            interventions=InterventionController(
                (
                    Intervention(
                        "bad-key",
                        InterventionTarget.TENSORDICT,
                        InterventionTiming.INPUT,
                        replacement=torch.ones(2, 2),
                        key="missing",
                    ),
                )
            ),
        )
