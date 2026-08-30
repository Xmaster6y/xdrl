import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow

from xdrl import (
    ArtifactDigestAlgorithm,
    BatchSemantics,
    InputArtifactReference,
    InputArtifactRole,
    InteractionContract,
    InteractionPhase,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    OutputArtifactDeclaration,
    OutputArtifactDigest,
    OutputArtifactRole,
    ProvenanceSchemaError,
    RuntimeInteractionContext,
    TDHookWorkflowRunner,
    TensorDictSchema,
    WorkflowProvenance,
)


def _runner() -> TDHookWorkflowRunner:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",)))
    contract = InteractionContract(
        "policy:evaluation:artifacts",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        inputs,
        outputs,
    )
    policy = TensorDictModule(torch.nn.Linear(2, 1), in_keys=["observation"], out_keys=["action"])
    batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
    return TDHookWorkflowRunner(RuntimeInteractionContext(contract, policy, batch))


def _input(identity: str, role: InputArtifactRole) -> InputArtifactReference:
    return InputArtifactReference(
        identity=identity,
        role=role,
        digest_algorithm=ArtifactDigestAlgorithm.SHA256,
        digest_value="a" * 64,
        source="https://example.invalid/research-assets.git",
        revision="0123456789abcdef",
        metadata={"license": "research-only", "fold": 2},
    )


@pytest.mark.integration
def test_workflow_records_typed_input_and_output_artifacts_deterministically() -> None:
    runner = _runner()
    inputs = (
        _input("checkpoint:policy-v1", InputArtifactRole.MODEL_CHECKPOINT),
        _input("split:evaluation-v2", InputArtifactRole.EVALUATION_SPLIT),
        _input("result:paper-reference", InputArtifactRole.REFERENCE_RESULT),
    )
    outputs = (
        OutputArtifactDeclaration(
            identity="metrics:run-17",
            role=OutputArtifactRole.METRICS_BUNDLE,
            source="s3://immutable-results/run-17/metrics.json",
            source_is_immutable=True,
            metadata={"format": "json"},
        ),
    )
    resolved_digests: list[str] = []

    def resolve_outputs(
        data: TensorDict, declarations: tuple[OutputArtifactDeclaration, ...]
    ) -> tuple[OutputArtifactDigest, ...]:
        assert data.get("action") is not None
        digest = hashlib.sha512(data.get("action").detach().numpy().tobytes()).hexdigest()
        resolved_digests.append(digest)
        return (OutputArtifactDigest(declarations[0].identity, ArtifactDigestAlgorithm.SHA512, digest),)

    execution = runner.run(
        Workflow(ActivationCaching("module")),
        runner.interaction.representative_input.clone(),
        code_revision="test-revision",
        input_artifacts=inputs,
        output_artifacts=outputs,
        output_artifact_resolver=resolve_outputs,
    )

    provenance = execution.provenance
    restored = WorkflowProvenance.from_json(provenance.to_json())
    assert restored == provenance
    assert restored.to_json() == provenance.to_json()
    assert [item.role for item in restored.input_artifacts] == [
        InputArtifactRole.MODEL_CHECKPOINT,
        InputArtifactRole.EVALUATION_SPLIT,
        InputArtifactRole.REFERENCE_RESULT,
    ]
    assert restored.output_artifacts[0].role is OutputArtifactRole.METRICS_BUNDLE
    assert restored.output_artifacts[0].digest_value == resolved_digests[0]

    with pytest.raises(TypeError):
        restored.input_artifacts[0].metadata["fold"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        restored.output_artifacts[0].identity = "changed"  # type: ignore[misc]


@pytest.mark.integration
def test_artifact_references_are_execution_evidence_not_a_reproduction_verdict() -> None:
    runner = _runner()
    execution = runner.run(
        Workflow(ActivationCaching("module")),
        runner.interaction.representative_input.clone(),
        code_revision="test-revision",
        input_artifacts=(_input("result:paper-reference", InputArtifactRole.REFERENCE_RESULT),),
    )

    assert execution.provenance.input_artifacts[0].role is InputArtifactRole.REFERENCE_RESULT
    assert not hasattr(execution.provenance, "result_reproduced")


@pytest.mark.integration
def test_artifact_validation_fails_before_workflow_execution() -> None:
    runner = _runner()
    duplicate_input = _input("artifact:duplicate", InputArtifactRole.DATASET)
    duplicate_output = OutputArtifactDeclaration(
        identity="artifact:duplicate",
        role=OutputArtifactRole.OTHER,
    )

    with pytest.raises(ProvenanceSchemaError, match="artifact identities must be unique"):
        runner.run(
            Workflow(ActivationCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
            input_artifacts=(duplicate_input,),
            output_artifacts=(duplicate_output,),
            output_artifact_resolver=lambda _data, _declarations: (),
        )

    assert not runner.interaction.events


@pytest.mark.integration
def test_output_artifact_resolver_must_match_declarations_after_execution() -> None:
    runner = _runner()
    declaration = OutputArtifactDeclaration("metrics:run-17", OutputArtifactRole.METRICS_BUNDLE)

    with pytest.raises(ProvenanceSchemaError, match="missing identities: metrics:run-17"):
        runner.run(
            Workflow(ActivationCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
            output_artifacts=(declaration,),
            output_artifact_resolver=lambda _data, _declarations: (),
        )

    assert [event.kind.value for event in runner.interaction.events] == ["before", "after"]


@pytest.mark.integration
def test_output_artifact_declaration_requires_a_resolver_before_execution() -> None:
    runner = _runner()
    declaration = OutputArtifactDeclaration("metrics:run-17", OutputArtifactRole.METRICS_BUNDLE)

    with pytest.raises(ValueError, match="declarations and output_artifact_resolver must be provided together"):
        runner.run(
            Workflow(ActivationCaching("module")),
            runner.interaction.representative_input.clone(),
            code_revision="test-revision",
            output_artifacts=(declaration,),
        )

    assert not runner.interaction.events


@pytest.mark.integration
def test_artifact_reference_can_be_replaced_after_metadata_is_frozen() -> None:
    original = _input("dataset:evaluation", InputArtifactRole.DATASET)

    updated = replace(original, identity="dataset:evaluation-v2")

    assert updated.metadata == original.metadata
    assert updated.identity == "dataset:evaluation-v2"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"identity": ""}, "identity must be a non-empty string"),
        ({"digest_value": "ABC"}, "digest_value must be 64 lowercase hexadecimal characters"),
        (
            {"source": "https://example.invalid/assets/main", "revision": None},
            "source requires a revision unless it is explicitly immutable",
        ),
        ({"metadata": {"callback": object()}}, "metadata must contain only JSON-compatible values"),
    ],
)
def test_input_artifact_reference_rejects_malformed_evidence(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "identity": "dataset:evaluation",
        "role": InputArtifactRole.DATASET,
        "digest_algorithm": ArtifactDigestAlgorithm.SHA256,
        "digest_value": "d" * 64,
    }
    values.update(kwargs)

    with pytest.raises(ProvenanceSchemaError, match=message):
        InputArtifactReference(**values)  # type: ignore[arg-type]


@pytest.mark.integration
def test_provenance_explicitly_rejects_revision_two_without_implicit_migration() -> None:
    runner = _runner()
    provenance = runner.run(
        Workflow(ActivationCaching("module")),
        runner.interaction.representative_input.clone(),
        code_revision="test-revision",
    ).provenance
    payload = provenance.to_dict()
    payload["schema_revision"] = 2
    del payload["input_artifacts"]
    del payload["output_artifacts"]

    with pytest.raises(ProvenanceSchemaError, match="cannot be migrated without caller-supplied artifact identities"):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
def test_artifact_decoder_rejects_unknown_fields_and_non_array_collections() -> None:
    reference = _input("dataset:evaluation", InputArtifactRole.DATASET).to_dict()
    reference["invented"] = True
    with pytest.raises(ProvenanceSchemaError, match="missing or unknown fields"):
        InputArtifactReference.from_dict(reference)

    runner = _runner()
    payload = runner.run(
        Workflow(ActivationCaching("module")),
        runner.interaction.representative_input.clone(),
        code_revision="test-revision",
    ).provenance.to_dict()
    payload["input_artifacts"] = {}
    with pytest.raises(ProvenanceSchemaError, match="input_artifacts must be an array"):
        WorkflowProvenance.from_dict(payload)
