from dataclasses import replace
import re

import pytest

from xdrl.interactions import InteractionDescriptor, InteractionPhase, SchemaSnapshot
from xdrl.provenance import PROVENANCE_SCHEMA_REVISION, ProvenanceManifest, ProvenanceSchemaError
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _descriptor() -> InteractionDescriptor:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",)))
    return InteractionDescriptor(
        identity="policy:evaluation:0",
        role=ModelRole.ACTOR,
        phase=InteractionPhase.EVALUATION,
        module_path="policy",
        input_schema=SchemaSnapshot.from_schema(inputs),
        output_schema=SchemaSnapshot.from_schema(outputs),
        batch_dimensions=("env",),
        exploration_mode="deterministic",
        gradient_enabled=False,
        model_id="actor-v2",
        checkpoint_id="sha256:abc",
    )


def _dependencies() -> dict[str, str]:
    return {
        "python": "3.11.0",
        "torch": "2.11.0",
        "tensordict": "0.12.2",
        "torchrl": "0.12.0+g5b2bc08b",
        "tdhook": "0.1.3",
        "xdrl": "0.1.0",
    }


@pytest.mark.integration
def test_provenance_round_trip_covers_reproduction_boundary() -> None:
    manifest = ProvenanceManifest.capture(
        _descriptor(),
        selected_keys=(("observation",), ("action",)),
        target_paths={"encoder": "td_module.module.0"},
        tdhook_method={"name": "activation_caching", "pattern": "module\\.0"},
        dependencies=_dependencies(),
        code_revision="6b9279a",
    )

    restored = ProvenanceManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.interaction["identity"] == "policy:evaluation:0"
    assert restored.checkpoint_id == "sha256:abc"
    assert restored.exploration_mode == "deterministic"
    assert restored.batch_dimensions == ("env",)


@pytest.mark.integration
def test_provenance_rejects_incompatible_schema_revisions() -> None:
    manifest = ProvenanceManifest.capture(
        replace(_descriptor(), checkpoint_id=None),
        selected_keys=(("observation",),),
        target_paths={},
        tdhook_method={"name": "noop"},
        dependencies=_dependencies(),
        code_revision="6b9279a",
    ).to_dict()
    manifest["schema_revision"] = PROVENANCE_SCHEMA_REVISION + 1

    with pytest.raises(ProvenanceSchemaError, match="unsupported provenance schema revision"):
        ProvenanceManifest.from_dict(manifest)


@pytest.mark.integration
def test_provenance_rejects_non_serialisable_method_configuration() -> None:
    with pytest.raises(ProvenanceSchemaError, match="non-serialisable"):
        ProvenanceManifest.capture(
            _descriptor(),
            selected_keys=(("observation",),),
            target_paths={},
            tdhook_method={"callback": object()},
            dependencies=_dependencies(),
            code_revision="6b9279a",
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gradient_enabled", "false", "gradient_enabled must be a boolean"),
        ("selected_keys", ["observation"], "selected_keys[0] must be a non-empty array"),
        ("target_paths", [], "target_paths must be an object"),
        ("tdhook_method", [], "tdhook_method must be an object"),
        ("dependencies", {**_dependencies(), "torch": ""}, "dependencies.torch must be a non-empty string"),
        ("dependencies", {**_dependencies(), "torch": "not-a-version"}, "dependencies.torch must be a valid version"),
        ("model_id", 7, "model_id must be a non-empty string"),
    ],
)
def test_provenance_rejects_malformed_field_types(field: str, value: object, message: str) -> None:
    manifest = ProvenanceManifest.capture(
        _descriptor(),
        selected_keys=(("observation",),),
        target_paths={},
        tdhook_method={"name": "noop"},
        dependencies=_dependencies(),
        code_revision="6b9279a",
    ).to_dict()
    manifest[field] = value

    with pytest.raises(ProvenanceSchemaError, match=re.escape(message)):
        ProvenanceManifest.from_dict(manifest)
