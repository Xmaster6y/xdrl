from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching, SteeringVectors
from tdhook.weights import Pruning
from tdhook.workflow import Workflow, WorkflowUpdate

from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext
from xdrl.provenance import (
    ArtifactDigestAlgorithm,
    InputArtifactReference,
    InputArtifactRole,
    OutputArtifactDeclaration,
    OutputArtifactDigest,
    OutputArtifactRole,
)
from xdrl.tdhook import (
    PairedWorkflowExecutionError,
    PairedWorkflowValidationError,
    TDHookWorkflowPairManifest,
    TDHookWorkflowRunner,
    WorkflowArmReference,
    WorkflowStepDifference,
)
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _runner(*, stochastic: bool = False, layer: torch.nn.Module | None = None) -> TDHookWorkflowRunner:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
        BatchSemantics(("env",)),
    )
    if layer is None:
        if stochastic:
            layer = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(2, 1, bias=False))
        else:
            layer = torch.nn.Linear(2, 1, bias=False)
    linear = layer[-1] if isinstance(layer, torch.nn.Sequential) else layer
    assert isinstance(linear, torch.nn.Linear)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    policy = TensorDictModule(layer, in_keys=["observation"], out_keys=["action"])
    contract = InteractionContract(
        "policy:evaluation:pair",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        inputs,
        outputs,
        model_id="actor-v3",
        checkpoint_id="sha256:paired",
        module_training=stochastic,
    )
    batch = TensorDict({"observation": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, batch_size=[2])
    return TDHookWorkflowRunner(RuntimeInteractionContext(contract, policy, batch))


def _cache() -> Workflow:
    return Workflow(ActivationCaching("module", cache_key=("activations", "head")))


def _arm_reference() -> WorkflowArmReference:
    return WorkflowArmReference("a" * 64, (), ())


def _manifest(**overrides: object) -> TDHookWorkflowPairManifest:
    values = {
        "schema_revision": 1,
        "pair_id": "pair:test",
        "interaction_id": "interaction:test",
        "model_id": "model:test",
        "checkpoint_id": "checkpoint:test",
        "changed_steps": (),
        "baseline": _arm_reference(),
        "intervention": _arm_reference(),
        "matched_rng_sources": ("torch_cpu",),
        "unsupported_randomness_sources": ("python_random",),
    }
    values.update(overrides)
    return TDHookWorkflowPairManifest(**values)  # type: ignore[arg-type]


def _resolve_artifacts(
    _data: TensorDict, declarations: tuple[OutputArtifactDeclaration, ...]
) -> tuple[OutputArtifactDigest, ...]:
    return tuple(
        OutputArtifactDigest(item.identity, ArtifactDigestAlgorithm.SHA256, "2" * 64) for item in declarations
    )


@pytest.mark.integration
def test_no_op_pair_matches_rng_and_returns_native_results_with_tensor_free_manifest() -> None:
    runner = _runner(stochastic=True)
    data = runner.interaction.representative_input
    torch.manual_seed(7)
    initial_rng = torch.get_rng_state().clone()
    runner.run(_cache(), data.clone(), code_revision="test-revision")
    expected_final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    checkpoint = InputArtifactReference(
        "checkpoint:actor-v3",
        InputArtifactRole.MODEL_CHECKPOINT,
        ArtifactDigestAlgorithm.SHA256,
        "1" * 64,
    )
    baseline_artifact = OutputArtifactDeclaration("result:baseline", OutputArtifactRole.INTERVENTION_RESULT)
    intervention_artifact = OutputArtifactDeclaration("result:intervention", OutputArtifactRole.INTERVENTION_RESULT)

    pair = runner.run_paired(
        _cache(),
        _cache(),
        data,
        pair_id="pair:no-op",
        code_revision="test-revision",
        input_artifacts=(checkpoint,),
        baseline_output_artifacts=(baseline_artifact,),
        baseline_output_artifact_resolver=_resolve_artifacts,
        intervention_output_artifacts=(intervention_artifact,),
        intervention_output_artifact_resolver=_resolve_artifacts,
    )

    torch.testing.assert_close(pair.baseline.data["action"], pair.intervention.data["action"])
    assert pair.baseline.provenance.model_calls == pair.intervention.provenance.model_calls == 1
    assert torch.equal(torch.get_rng_state(), expected_final_rng)
    assert pair.manifest.changed_steps == ()
    assert pair.manifest.matched_rng_sources[0] == "torch_cpu"
    assert "python_random" in pair.manifest.unsupported_randomness_sources
    assert pair.manifest.interpretation == "mechanics_and_provenance_only"
    assert len(pair.manifest.baseline.provenance_sha256) == 64
    assert pair.manifest.baseline.input_artifacts == ("checkpoint:actor-v3",)
    assert pair.manifest.baseline.output_artifacts == ("result:baseline",)
    assert pair.manifest.intervention.output_artifacts == ("result:intervention",)
    assert TDHookWorkflowPairManifest.from_json(pair.manifest.to_json()) == pair.manifest
    assert "Tensor" not in pair.manifest.to_json()
    assert "action" not in data.keys()
    assert pair.baseline.data is not pair.intervention.data


def _keep_intervention_output(*, output: torch.Tensor, **_: object) -> torch.Tensor:
    return output


def _zero_intervention_output(*, output: torch.Tensor, **_: object) -> torch.Tensor:
    return torch.zeros_like(output)


@pytest.mark.integration
def test_pair_records_a_declared_nontrivial_activation_intervention() -> None:
    runner = _runner()
    data = runner.interaction.representative_input.clone()
    baseline = Workflow(SteeringVectors(["module"], steer_fn=_keep_intervention_output))
    intervention = Workflow(SteeringVectors(["module"], steer_fn=_zero_intervention_output))

    pair = runner.run_paired(
        baseline,
        intervention,
        data,
        pair_id="pair:activation",
        code_revision="test-revision",
        declared_workflow_differences=(0,),
        callback_identifiers={
            _keep_intervention_output: "keep_activation",
            _zero_intervention_output: "zero_activation",
        },
    )

    assert tuple(item.index for item in pair.manifest.changed_steps) == (0,)
    assert not torch.equal(pair.baseline.data["action"], pair.intervention.data["action"])
    assert not any(child._forward_hooks for child in runner.interaction.module.modules())


def _importance(*, parameter: torch.Tensor, **_: object) -> torch.Tensor:
    return parameter.abs()


@pytest.mark.integration
def test_parameter_intervention_is_applied_only_to_its_arm_and_restored() -> None:
    runner = _runner()
    original = {name: value.detach().clone() for name, value in runner.interaction.module.state_dict().items()}
    baseline = Workflow(Pruning(importance_callback=_importance, amount_to_prune=0, relative_path="module"))
    intervention = Workflow(Pruning(importance_callback=_importance, amount_to_prune=1, relative_path="module"))

    pair = runner.run_paired(
        baseline,
        intervention,
        runner.interaction.representative_input,
        pair_id="pair:parameter",
        code_revision="test-revision",
        declared_workflow_differences=(0,),
        callback_identifiers={_importance: "absolute_parameter_importance"},
    )

    assert torch.count_nonzero(pair.baseline.data["action"])
    assert not torch.equal(pair.baseline.data["action"], pair.intervention.data["action"])
    torch.testing.assert_close(pair.intervention.data["action"], torch.tensor([[2.0], [6.0]]))
    for name, value in runner.interaction.module.state_dict().items():
        torch.testing.assert_close(value, original[name])


class _FailingWorkflow(Workflow):
    def run_with_plan(self, model: torch.nn.Module, data: TensorDict):  # type: ignore[no-untyped-def]
        super().run_with_plan(model, data)
        raise RuntimeError("arm exploded")


@pytest.mark.integration
@pytest.mark.parametrize("failed_arm", ["baseline", "intervention"])
def test_failure_in_either_arm_restores_module_rng_and_hooks(failed_arm: str) -> None:
    runner = _runner(stochastic=True)
    good = _cache()
    failed = _FailingWorkflow(ActivationCaching("module", cache_key=("activations", "head")))
    workflows = (failed, good) if failed_arm == "baseline" else (good, failed)
    original = {name: value.detach().clone() for name, value in runner.interaction.module.state_dict().items()}
    torch.manual_seed(17)
    initial_rng = torch.get_rng_state().clone()

    with pytest.raises(PairedWorkflowExecutionError, match=failed_arm) as caught:
        runner.run_paired(
            *workflows,
            runner.interaction.representative_input,
            pair_id=f"pair:failure:{failed_arm}",
            code_revision="test-revision",
        )

    assert caught.value.arm == failed_arm
    assert (caught.value.baseline is not None) is (failed_arm == "intervention")
    expected_rng = initial_rng
    if failed_arm == "intervention":
        torch.set_rng_state(initial_rng)
        runner.run(good, runner.interaction.representative_input.clone(), code_revision="test-revision")
        expected_rng = torch.get_rng_state().clone()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in runner.interaction.module.state_dict().items():
        torch.testing.assert_close(value, original[name])
    assert not any(child._forward_hooks for child in runner.interaction.module.modules())


@pytest.mark.integration
def test_pair_rejects_undeclared_differences_and_incompatible_contracts_before_execution() -> None:
    runner = _runner()
    baseline = Workflow(Pruning(importance_callback=_importance, amount_to_prune=0, relative_path="module"))
    intervention = Workflow(Pruning(importance_callback=_importance, amount_to_prune=1, relative_path="module"))

    with pytest.raises(PairedWorkflowValidationError, match="do not match declaration"):
        runner.run_paired(
            baseline,
            intervention,
            runner.interaction.representative_input,
            pair_id="pair:undeclared",
            code_revision="test-revision",
            callback_identifiers={_importance: "absolute_parameter_importance"},
        )

    other_contract = replace(runner.interaction.contract, checkpoint_id="sha256:different")
    other = TDHookWorkflowRunner(
        RuntimeInteractionContext(
            other_contract,
            runner.interaction.module,
            runner.interaction.representative_input,
        )
    )
    with pytest.raises(PairedWorkflowValidationError, match="interaction, model, and checkpoint"):
        runner.run_paired(
            _cache(),
            _cache(),
            runner.interaction.representative_input,
            pair_id="pair:incompatible",
            code_revision="test-revision",
            intervention_runner=other,
        )

    assert not runner.interaction.events


@pytest.mark.integration
@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WorkflowStepDifference(-1, "baseline", "intervention"), "non-negative"),
        (lambda: WorkflowStepDifference(0, "same", "same"), "distinct"),
        (lambda: WorkflowArmReference("invalid", (), ()), "SHA-256"),
        (lambda: WorkflowArmReference("a" * 64, [], ()), "tuple of non-empty"),  # type: ignore[arg-type]
        (lambda: WorkflowArmReference("a" * 64, ("duplicate", "duplicate"), ()), "unique"),
        (lambda: _manifest(schema_revision=2), "schema revision"),
        (lambda: _manifest(pair_id=""), "non-empty string"),
        (
            lambda: _manifest(
                changed_steps=(
                    WorkflowStepDifference(1, "a", "b"),
                    WorkflowStepDifference(0, "c", "d"),
                )
            ),
            "unique and sorted",
        ),
        (lambda: _manifest(changed_steps=(object(),)), "WorkflowStepDifference"),
        (lambda: _manifest(baseline=object()), "WorkflowArmReference"),
        (lambda: _manifest(matched_rng_sources=[]), "tuple of non-empty"),
        (lambda: _manifest(matched_rng_sources=("torch_cpu", "torch_cpu")), "unique"),
        (lambda: _manifest(interpretation="causal"), "cannot assign a causal"),
    ],
)
def test_pair_manifest_rejects_invalid_values(factory: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be an object"),
        ({}, "missing or unknown fields"),
        ({**_manifest().to_dict(), "changed_steps": "invalid"}, "changed_steps must be an array"),
        ({**_manifest().to_dict(), "baseline": {}}, "baseline arm reference"),
        (
            {
                **_manifest().to_dict(),
                "baseline": {**_manifest().to_dict()["baseline"], "input_artifacts": "artifact"},
            },
            "baseline.input_artifacts must be an array",
        ),
        ({**_manifest().to_dict(), "matched_rng_sources": "torch_cpu"}, "matched_rng_sources must be an array"),
    ],
)
def test_pair_manifest_decoder_rejects_invalid_payloads(payload: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TDHookWorkflowPairManifest.from_dict(payload)  # type: ignore[arg-type]


@pytest.mark.integration
def test_pair_manifest_json_decoder_requires_an_object() -> None:
    with pytest.raises(TypeError, match="JSON must contain an object"):
        TDHookWorkflowPairManifest.from_json("[]")


@pytest.mark.integration
@pytest.mark.parametrize("declared", [[], (0, 0)])
def test_pair_rejects_invalid_difference_declarations(declared: object) -> None:
    runner = _runner()
    with pytest.raises(PairedWorkflowValidationError, match="tuple of non-negative|unique and sorted"):
        runner.run_paired(
            _cache(),
            _cache(),
            runner.interaction.representative_input,
            pair_id="pair:invalid-differences",
            code_revision="test-revision",
            declared_workflow_differences=declared,  # type: ignore[arg-type]
        )


@pytest.mark.integration
def test_pair_rejects_missing_identity_contract_drift_and_a_different_module() -> None:
    runner = _runner()
    data = runner.interaction.representative_input

    with pytest.raises(PairedWorkflowValidationError, match="pair_id"):
        runner.run_paired(_cache(), _cache(), data, pair_id="", code_revision="test-revision")

    missing_identity = TDHookWorkflowRunner(
        RuntimeInteractionContext(
            replace(runner.interaction.contract, model_id=None),
            runner.interaction.module,
            data,
        )
    )
    with pytest.raises(PairedWorkflowValidationError, match="explicit model and checkpoint"):
        missing_identity.run_paired(_cache(), _cache(), data, pair_id="pair:missing", code_revision="test-revision")

    drifted = TDHookWorkflowRunner(
        RuntimeInteractionContext(
            replace(runner.interaction.contract, module_training=True),
            runner.interaction.module,
            data,
        )
    )
    with pytest.raises(PairedWorkflowValidationError, match="identical interaction contracts"):
        runner.run_paired(
            _cache(),
            _cache(),
            data,
            pair_id="pair:drift",
            code_revision="test-revision",
            intervention_runner=drifted,
        )

    different_module = _runner()
    with pytest.raises(PairedWorkflowValidationError, match="same module instance"):
        runner.run_paired(
            _cache(),
            _cache(),
            data,
            pair_id="pair:different-module",
            code_revision="test-revision",
            intervention_runner=different_module,
        )

    separate_context = TDHookWorkflowRunner(
        RuntimeInteractionContext(
            runner.interaction.contract,
            runner.interaction.module,
            data,
        )
    )
    with pytest.raises(PairedWorkflowValidationError, match="same runtime interaction context"):
        runner.run_paired(
            _cache(),
            _cache(),
            data,
            pair_id="pair:different-context",
            code_revision="test-revision",
            intervention_runner=separate_context,
        )


@pytest.mark.integration
def test_pair_rejects_unshared_tensordict_operators() -> None:
    runner = _runner()
    baseline_operator = TensorDictModule(torch.nn.Identity(), in_keys=["observation"], out_keys=["observation"])
    intervention_operator = TensorDictModule(torch.nn.Identity(), in_keys=["observation"], out_keys=["observation"])

    with pytest.raises(PairedWorkflowValidationError, match="same instances"):
        runner.run_paired(
            Workflow(baseline_operator, ActivationCaching("module")),
            Workflow(intervention_operator, ActivationCaching("module")),
            runner.interaction.representative_input,
            pair_id="pair:operators",
            code_revision="test-revision",
        )


class _MutatingLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1, bias=False)
        self.register_buffer("counter", torch.zeros((), dtype=torch.int64), persistent=False)
        self.seen: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.counter.add_(1)
        self.seen.append(int(self.counter))
        return super().forward(value)


class _MutatingIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("counter", torch.zeros((), dtype=torch.int64), persistent=False)
        self.seen: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.counter.add_(1)
        self.seen.append(int(self.counter))
        return value


@pytest.mark.integration
def test_pair_restores_nonpersistent_interaction_buffers_between_arms_and_afterward() -> None:
    layer = _MutatingLinear()
    runner = _runner(layer=layer)

    pair = runner.run_paired(
        _cache(),
        _cache(),
        runner.interaction.representative_input,
        pair_id="pair:nonpersistent-buffer",
        code_revision="test-revision",
    )

    torch.testing.assert_close(pair.baseline.data["action"], pair.intervention.data["action"])
    assert layer.seen == [1, 1]
    assert not layer.counter


@pytest.mark.integration
def test_pair_restores_shared_operator_state_between_arms_and_afterward() -> None:
    runner = _runner()
    layer = _MutatingIdentity()
    operator = TensorDictModule(layer, in_keys=["observation"], out_keys=["observation"])
    baseline = Workflow(WorkflowUpdate(operator), ActivationCaching("module"))
    intervention = Workflow(WorkflowUpdate(operator), ActivationCaching("module"))

    pair = runner.run_paired(
        baseline,
        intervention,
        runner.interaction.representative_input,
        pair_id="pair:stateful-operator",
        code_revision="test-revision",
    )

    torch.testing.assert_close(pair.baseline.data["action"], pair.intervention.data["action"])
    assert layer.seen == [1, 1]
    assert not layer.counter
