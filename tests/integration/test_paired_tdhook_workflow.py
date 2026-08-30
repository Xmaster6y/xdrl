from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching, SteeringVectors
from tdhook.weights import Pruning
from tdhook.workflow import Workflow

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
)
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _runner(*, stochastic: bool = False) -> TDHookWorkflowRunner:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
        BatchSemantics(("env",)),
    )
    layer: torch.nn.Module
    if stochastic:
        layer = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(2, 1, bias=False))
    else:
        layer = torch.nn.Linear(2, 1, bias=False)
    linear = layer[-1] if stochastic else layer
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
