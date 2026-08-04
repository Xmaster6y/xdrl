import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookHandle
from tdhook.modules import HookedModule
from tdhook.pipeline import MethodStage, Pipeline, TransformStage

from xdrl.interactions import (
    InteractionDescriptor,
    InteractionPhase,
    LifecycleEventType,
    RuntimeInteractionContext,
    SchemaSnapshot,
)
from xdrl.tdhook import TDHookInteractionAdapter
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


INPUT_KEY = ("inputs", "input")
OUTPUT_KEY = ("outputs", "model")


class AddOne(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return module.register_submodule_hook("module", lambda module, args, output: output + 1, direction="fwd")


class TimesTwo(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return module.register_submodule_hook("module", lambda module, args, output: output * 2, direction="fwd")


class CountingLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1, bias=False)
        self.calls = 0
        self.training_during_calls: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.training_during_calls.append(self.training)
        return super().forward(value)


def _interaction() -> tuple[RuntimeInteractionContext, CountingLinear]:
    inputs = TensorDictSchema(
        (KeySchema(INPUT_KEY, KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (KeySchema(OUTPUT_KEY, KeyRole.ACTION, KeyPresence.PRODUCED),),
        BatchSemantics(("env",)),
    )
    layer = CountingLinear()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0]]))
    policy = TensorDictModule(layer, in_keys=[INPUT_KEY], out_keys=[OUTPUT_KEY])
    policy.train()
    batch = TensorDict({"inputs": {"input": torch.ones(3, 2)}}, batch_size=[3])
    descriptor = InteractionDescriptor(
        identity="policy:evaluation:planned",
        role=ModelRole.ACTOR,
        phase=InteractionPhase.EVALUATION,
        module_path="policy",
        input_schema=SchemaSnapshot.from_schema(inputs),
        output_schema=SchemaSnapshot.from_schema(outputs),
        batch_dimensions=("env",),
        model_id="actor-v3",
        checkpoint_id="sha256:planned",
        module_training=False,
    )
    return RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch), layer


def _method(
    name: str,
    factory: HookingContextFactory,
    *,
    required_keys: tuple[tuple[str, str], ...],
    provided_keys: tuple[tuple[str, str], ...] = (),
    coexecution_key: str | None = None,
) -> MethodStage:
    return MethodStage(
        name,
        factory,
        required_keys=required_keys,
        provided_keys=provided_keys,
        model_in_keys=[INPUT_KEY],
        model_out_keys=[OUTPUT_KEY],
        coexecution_key=coexecution_key,
        device_batch_constraints=["same interaction batch"] if coexecution_key else (),
    )


@pytest.mark.integration
def test_planned_compatible_stages_share_one_validated_interaction_call() -> None:
    interaction, layer = _interaction()
    artifacts = interaction.representative_input.clone()
    pipeline = Pipeline(
        [
            _method("add", AddOne(), required_keys=(INPUT_KEY,), coexecution_key="ordered-hooks-v1"),
            _method(
                "multiply",
                TimesTwo(),
                required_keys=(INPUT_KEY,),
                provided_keys=(OUTPUT_KEY,),
                coexecution_key="ordered-hooks-v1",
            ),
        ]
    )

    result = TDHookInteractionAdapter(interaction).run_pipeline(pipeline, artifacts, code_revision="abc123", seed=7)

    assert result.plan.model_passes == 1
    assert result.plan.runs[0].coalesced
    assert layer.calls == 1
    assert layer.training_during_calls == [False]
    assert interaction.module.training
    assert torch.equal(result.artifacts.get(OUTPUT_KEY), torch.full((3, 1), 4.0))
    assert [event.kind for event in interaction.events] == [LifecycleEventType.BEFORE, LifecycleEventType.AFTER]
    assert len(result.interaction_provenance) == 2
    manifest = result.interaction_provenance[0]
    assert manifest.model_id == "actor-v3"
    assert manifest.checkpoint_id == "sha256:planned"
    assert manifest.tdhook_method["planned_run"]["model_passes"] == 1
    assert manifest.tdhook_method["planned_run"]["coalesced"] is True
    assert not layer._forward_hooks


@pytest.mark.integration
def test_planned_incompatible_stages_remain_separate_and_report_two_passes() -> None:
    interaction, layer = _interaction()
    pipeline = Pipeline(
        [
            _method("first", AddOne(), required_keys=(INPUT_KEY,), provided_keys=(OUTPUT_KEY,)),
            _method("second", TimesTwo(), required_keys=(OUTPUT_KEY,)),
        ]
    )

    result = TDHookInteractionAdapter(interaction).run_pipeline(
        pipeline, interaction.representative_input.clone(), code_revision="abc123"
    )

    assert [run.stages for run in result.plan.runs] == [("first",), ("second",)]
    assert result.plan.model_passes == 2
    assert layer.calls == 2
    assert [event.kind for event in interaction.events] == [
        LifecycleEventType.BEFORE,
        LifecycleEventType.AFTER,
        LifecycleEventType.BEFORE,
        LifecycleEventType.AFTER,
    ]
    assert not layer._forward_hooks


@pytest.mark.integration
def test_every_planned_call_revalidates_schema_and_restores_hooks_after_failure() -> None:
    interaction, layer = _interaction()
    pipeline = Pipeline(
        [
            _method("first", AddOne(), required_keys=(INPUT_KEY,), provided_keys=(OUTPUT_KEY,)),
            TransformStage("drop-input", lambda td: td.del_(INPUT_KEY), required_keys=[OUTPUT_KEY]),
            _method("second", TimesTwo(), required_keys=(OUTPUT_KEY,)),
        ]
    )

    with pytest.raises(RuntimeError, match="second.*missing required key"):
        TDHookInteractionAdapter(interaction).run_pipeline(
            pipeline, interaction.representative_input.clone(), code_revision="abc123"
        )

    assert layer.calls == 1
    assert interaction.module.training
    assert [event.kind for event in interaction.events] == [
        LifecycleEventType.BEFORE,
        LifecycleEventType.AFTER,
        LifecycleEventType.BEFORE,
        LifecycleEventType.FAILURE,
    ]
    assert not layer._forward_hooks
