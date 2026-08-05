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
    InteractionContract,
    InteractionPhase,
    LifecycleEventType,
    RuntimeInteractionContext,
)
from xdrl.interventions import (
    Intervention,
    InterventionController,
    InterventionTarget,
    InterventionTiming,
    InterventionValidationError,
    run_paired,
)
from xdrl.observations import ObservationTrace, RetentionPolicy, TensorRetention
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


def _contract(
    phase: InteractionPhase, input_schema: TensorDictSchema, output_schema: TensorDictSchema
) -> InteractionContract:
    return InteractionContract(
        identity=f"policy:{phase.value}:7",
        role=ModelRole.ACTOR,
        phase=phase,
        module_path="policy.module",
        input_schema=input_schema,
        output_schema=output_schema,
        exploration_mode="random" if phase is InteractionPhase.COLLECTION else "deterministic",
        gradient_enabled=False,
        logical_step=7,
    )


def test_collection_and_evaluation_are_distinct_contexts_for_one_policy() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(3, 2)}, batch_size=[3])
    collection = RuntimeInteractionContext(_contract(InteractionPhase.COLLECTION, inputs, outputs), _policy(), batch)
    evaluation = RuntimeInteractionContext(_contract(InteractionPhase.EVALUATION, inputs, outputs), _policy(), batch)

    with collection:
        collection.invoke(batch.clone())
    with evaluation:
        evaluation.invoke(batch.clone())

    assert collection.contract.identity != evaluation.contract.identity
    assert collection.contract.exploration_mode != evaluation.contract.exploration_mode
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

    contract = _contract(InteractionPhase.COLLECTION, inputs, outputs)
    original_exploration = exploration_type()
    original_grad = torch.is_grad_enabled()
    context = RuntimeInteractionContext(contract, _policy(), batch, hook_state)
    with pytest.raises(RuntimeError, match="boom"):
        with context:
            assert exploration_type().value == "random"
            assert not torch.is_grad_enabled()
            raise RuntimeError("boom")
    assert exploration_type() == original_exploration
    assert torch.is_grad_enabled() == original_grad
    assert entered == ["entered", "exited"]


def test_one_shot_call_is_a_synchronous_policy_and_restores_model_mode() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(2, 2)}, batch_size=[2])
    policy = _policy()
    policy.train()
    contract = replace(
        _contract(InteractionPhase.EVALUATION, inputs, outputs),
        module_training=False,
    )
    observed_modes: list[bool] = []
    original_forward = policy.module.forward

    def record_mode(value: torch.Tensor) -> torch.Tensor:
        observed_modes.append(policy.training)
        return original_forward(value)

    policy.module.forward = record_mode
    context = RuntimeInteractionContext(contract, policy, batch)

    result = context(batch.clone())

    assert set(result.keys()) == {"observation", "action"}
    assert observed_modes == [False]
    assert policy.training
    assert [event.kind for event in context.events] == [LifecycleEventType.BEFORE, LifecycleEventType.AFTER]


def test_context_restores_mixed_submodule_modes_after_failure() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    policy = _policy()
    policy.train()
    policy.module.eval()
    original_modes = tuple(module.training for module in policy.modules())
    contract = replace(
        _contract(InteractionPhase.EVALUATION, inputs, outputs),
        module_training=False,
    )
    context = RuntimeInteractionContext(contract, policy, batch)

    with pytest.raises(RuntimeError, match="boom"):
        with context:
            assert all(not module.training for module in policy.modules())
            raise RuntimeError("boom")

    assert tuple(module.training for module in policy.modules()) == original_modes


def test_context_applies_and_restores_mode_on_an_invocation_override() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    wrapped_policy = _policy()
    override = _policy()
    wrapped_policy.train()
    override.train()
    observed_modes: list[bool] = []
    original_forward = override.module.forward

    def record_mode(value: torch.Tensor) -> torch.Tensor:
        observed_modes.append(override.training)
        return original_forward(value)

    override.module.forward = record_mode
    contract = replace(
        _contract(InteractionPhase.EVALUATION, inputs, outputs),
        module_training=False,
    )
    context = RuntimeInteractionContext(contract, wrapped_policy, batch)

    with context:
        context.invoke(batch.clone(), module=override)
        assert not wrapped_policy.training
        assert not override.training

    assert observed_modes == [False]
    assert wrapped_policy.training
    assert override.training


def test_context_enables_gradients_inside_outer_inference_mode() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    contract = _contract(InteractionPhase.OPTIMISATION, inputs, outputs)
    contract = replace(contract, gradient_enabled=True, inference_mode=False)
    context = RuntimeInteractionContext(contract, _policy(), batch)

    with torch.inference_mode():
        with context:
            assert torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
        assert torch.is_inference_mode_enabled()


def test_contract_serialisation_preserves_spec_constraints() -> None:
    inputs, _ = _schemas()
    lower = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED, Bounded(low=-1, high=1, shape=(2,))),),
        BatchSemantics(("env",)),
    )
    wider = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED, Bounded(low=-2, high=2, shape=(2,))),),
        BatchSemantics(("env",)),
    )

    lower_contract = InteractionContract(
        "policy:output:lower", ModelRole.ACTOR, InteractionPhase.EVALUATION, "policy.module", inputs, lower
    )
    wider_contract = replace(lower_contract, identity="policy:output:wider", output_schema=wider)
    lower_key = lower_contract.to_dict()["output_schema"]["keys"][0]
    wider_key = wider_contract.to_dict()["output_schema"]["keys"][0]

    assert lower_key["spec_type"] == wider_key["spec_type"]
    assert lower_key["feature_shape"] == wider_key["feature_shape"]
    assert lower_key["spec_constraints"] != wider_key["spec_constraints"]
    json.dumps(lower_contract.to_dict())


def test_contract_rejects_disagreeing_input_and_output_batch_semantics() -> None:
    inputs, outputs = _schemas()
    different_outputs = TensorDictSchema(
        (KeySchema("different_action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("time",))
    )

    with pytest.raises(ValueError, match="identical batch semantics"):
        InteractionContract(
            "policy:evaluation:mismatch",
            ModelRole.ACTOR,
            InteractionPhase.EVALUATION,
            "policy.module",
            inputs,
            different_outputs,
        )


def test_contract_owns_live_schemas_and_derives_batch_semantics() -> None:
    inputs, outputs = _schemas()
    contract = InteractionContract(
        "policy:collection:1",
        ModelRole.ACTOR,
        InteractionPhase.COLLECTION,
        "policy.module",
        inputs,
        outputs,
        environment="CartPole",
    )

    assert contract.batch_dimensions == ("env",)
    assert contract.input_schema is inputs
    assert contract.output_schema is outputs
    assert contract.environment == "CartPole"
    assert contract.model_id is None


class _FailingPolicy:
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        raise RuntimeError("policy failure")


class _InvalidOutputPolicy:
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return TensorDict({"unexpected": torch.zeros(*tensordict.batch_size, 3)}, batch_size=tensordict.batch_size)


def test_module_mode_requires_a_torch_module() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(1, 2)}, batch_size=[1])
    contract = replace(
        _contract(InteractionPhase.EVALUATION, inputs, outputs),
        module_training=False,
    )
    context = RuntimeInteractionContext(contract, _FailingPolicy(), batch)

    with pytest.raises(TypeError, match="invoked module"):
        with context:
            pass


def test_failure_event_retains_only_diagnostics() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.zeros(2, 2)}, batch_size=[2])
    context = RuntimeInteractionContext(
        _contract(InteractionPhase.EVALUATION, inputs, outputs), _FailingPolicy(), batch
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
        _contract(InteractionPhase.EVALUATION, inputs, outputs), _InvalidOutputPolicy(), batch
    )

    with context, pytest.raises(Exception, match="missing produced key"):
        context.invoke(batch)

    assert context.events[-1].kind is LifecycleEventType.FAILURE
    assert context.events[-1].key_shapes == {"unexpected": (2, 3)}


def test_tensordict_interventions_are_checked_and_record_checkpoint_provenance() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
    contract = replace(_contract(InteractionPhase.EVALUATION, inputs, outputs), checkpoint_id="checkpoint-9")
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
    baseline = RuntimeInteractionContext(contract, policy, batch)
    control = RuntimeInteractionContext(contract, policy, batch, interventions=InterventionController((no_op,)))
    steered = RuntimeInteractionContext(contract, policy, batch, interventions=InterventionController((changed,)))

    control_result = run_paired(baseline, control, batch)
    result = run_paired(baseline, steered, batch)
    expected = control_result.baseline.get("action")
    assert torch.equal(control_result.intervention.get("action"), expected)
    actual = result.intervention.get("action")

    assert torch.equal(actual, expected + 1)
    assert (result.interaction_id, result.checkpoint_id) == (contract.identity, "checkpoint-9")
    record = steered.interventions.records[0]
    assert (record.interaction_id, record.checkpoint_id) == (contract.identity, "checkpoint-9")


def test_paired_execution_reuses_randomness_for_a_no_op_intervention() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
    policy = TensorDictModule(torch.nn.Dropout(p=0.5), in_keys=["observation"], out_keys=["action"])
    contract = _contract(InteractionPhase.EVALUATION, inputs, outputs)
    no_op = Intervention(
        "control",
        InterventionTarget.TENSORDICT,
        InterventionTiming.OUTPUT,
        transform=lambda value: value.clone(),
        key="action",
    )
    baseline = RuntimeInteractionContext(contract, policy, batch)
    control = RuntimeInteractionContext(contract, policy, batch, interventions=InterventionController((no_op,)))

    torch.manual_seed(7)
    result = run_paired(baseline, control, batch)

    assert torch.equal(result.baseline.get("action"), result.intervention.get("action"))


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
        _contract(InteractionPhase.EVALUATION, inputs, outputs),
        _policy(),
        batch,
        interventions=InterventionController((invalid,)),
    )

    with context, pytest.raises(InterventionValidationError, match="must preserve shape"):
        context.invoke(batch.clone())

    with pytest.raises(InterventionValidationError, match="declared required key"):
        RuntimeInteractionContext(
            _contract(InteractionPhase.EVALUATION, inputs, outputs),
            _policy(),
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


def test_input_observations_capture_the_intervened_value() -> None:
    inputs, outputs = _schemas()
    batch = TensorDict({"observation": torch.ones(1, 2)}, batch_size=[1])
    intervention = Intervention(
        "zero-input",
        InterventionTarget.TENSORDICT,
        InterventionTiming.INPUT,
        transform=torch.zeros_like,
        key="observation",
    )
    trace = ObservationTrace(RetentionPolicy(tensor=TensorRetention.DETACHED))
    context = RuntimeInteractionContext(
        _contract(InteractionPhase.EVALUATION, inputs, outputs),
        _policy(),
        batch,
        observations=trace,
        interventions=InterventionController((intervention,)),
    )

    with context:
        context.invoke(batch.clone())

    assert torch.equal(trace.records[0].payload, torch.zeros(1, 2))
