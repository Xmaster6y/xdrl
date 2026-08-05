import re

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow

from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext
from xdrl.provenance import (
    WORKFLOW_PROVENANCE_SCHEMA_REVISION,
    ProvenanceSchemaError,
    WorkflowProvenance,
)
from xdrl.tdhook import TDHookWorkflowRunner
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _interaction() -> RuntimeInteractionContext:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",)))
    contract = InteractionContract(
        identity="policy:evaluation:provenance",
        role=ModelRole.ACTOR,
        phase=InteractionPhase.EVALUATION,
        module_path="policy",
        input_schema=inputs,
        output_schema=outputs,
        exploration_mode="deterministic",
        model_id="actor-v2",
        checkpoint_id="sha256:abc",
    )
    policy = TensorDictModule(torch.nn.Linear(2, 1), in_keys=["observation"], out_keys=["action"])
    batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
    return RuntimeInteractionContext(contract, policy, batch)


def _dependencies() -> dict[str, str]:
    return {
        "python": "3.11.0",
        "torch": "2.13.0",
        "tensordict": "0.13.0",
        "torchrl": "0.13.0",
        "tdhook": "0.2.0",
        "xdrl": "0.2.0",
    }


def _run() -> WorkflowProvenance:
    interaction = _interaction()
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))
    result = TDHookWorkflowRunner(interaction).run(
        workflow,
        interaction.representative_input.clone(),
        code_revision="6b9279a",
        seed=17,
        dependencies=_dependencies(),
    )
    return result.provenance


@pytest.mark.integration
def test_workflow_provenance_round_trip_covers_verified_execution_boundary() -> None:
    provenance = _run()

    restored = WorkflowProvenance.from_json(provenance.to_json())

    assert restored == provenance
    assert restored.interaction_id == "policy:evaluation:provenance"
    assert restored.interaction_contract["checkpoint_id"] == "sha256:abc"
    assert restored.workflow_plan.model_passes == restored.model_calls == 1
    assert restored.workflow_plan.executions[0].steps == ("0:ActivationCaching",)
    assert restored.seed == 17


@pytest.mark.integration
def test_workflow_provenance_rejects_unknown_revisions_and_fields() -> None:
    payload = _run().to_dict()
    payload["schema_revision"] = WORKFLOW_PROVENANCE_SCHEMA_REVISION + 1
    with pytest.raises(ProvenanceSchemaError, match="unsupported workflow provenance schema revision"):
        WorkflowProvenance.from_dict(payload)

    payload = _run().to_dict()
    payload["invented"] = True
    with pytest.raises(ProvenanceSchemaError, match="unknown fields"):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
def test_workflow_provenance_rejects_plan_to_event_disagreement() -> None:
    payload = _run().to_dict()
    payload["lifecycle_events"] = payload["lifecycle_events"][:1]

    with pytest.raises(ProvenanceSchemaError, match="plan and lifecycle evidence disagree"):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", "17", "seed must be an integer or null"),
        ("code_revision", "", "code_revision must be a non-empty string"),
        ("dependencies", {**_dependencies(), "torch": ""}, "dependencies.torch must be a non-empty string"),
        ("dependencies", {**_dependencies(), "torch": "bad"}, "dependencies.torch must be a valid version"),
        ("interaction_contract", [], "interaction_contract must be an object"),
    ],
)
def test_workflow_provenance_rejects_malformed_fields(field: str, value: object, message: str) -> None:
    payload = _run().to_dict()
    payload[field] = value

    with pytest.raises(ProvenanceSchemaError, match=re.escape(message)):
        WorkflowProvenance.from_dict(payload)
