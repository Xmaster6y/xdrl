import re
from collections.abc import Callable, Mapping
from dataclasses import replace

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
    assert restored.configured_steps
    assert restored.seed == 17


@pytest.mark.integration
def test_workflow_provenance_uses_the_optional_seed_default_when_decoding() -> None:
    payload = _run().to_dict()
    del payload["seed"]

    restored = WorkflowProvenance.from_dict(payload)

    assert restored.seed is None


@pytest.mark.integration
def test_workflow_provenance_rejects_tuple_configured_steps_when_decoding() -> None:
    payload = _run().to_dict()
    payload["configured_steps"] = ("not JSON",)

    with pytest.raises(ProvenanceSchemaError, match="configured_steps must be an array"):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
def test_workflow_provenance_is_deeply_immutable_and_returns_detached_payloads() -> None:
    provenance = _run()
    contract = provenance.interaction_contract
    schema = contract["input_schema"]
    assert isinstance(schema, Mapping)

    with pytest.raises(TypeError):
        provenance.dependencies["torch"] = "0.0"  # type: ignore[index]
    with pytest.raises(TypeError):
        contract["role"] = "critic"  # type: ignore[index]
    with pytest.raises(TypeError):
        schema["keys"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        provenance.lifecycle_events[0].key_shapes["observation"] = (99,)  # type: ignore[index]

    payload = provenance.to_dict()
    payload["interaction_contract"]["role"] = "critic"
    assert provenance.interaction_contract["role"] == "actor"
    assert replace(provenance, seed=18).seed == 18


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
        ("configured_steps", [""], "configured_steps[0] must be a non-empty string"),
        ("interaction_contract", [], "interaction_contract must be an object"),
    ],
)
def test_workflow_provenance_rejects_malformed_fields(field: str, value: object, message: str) -> None:
    payload = _run().to_dict()
    payload[field] = value

    with pytest.raises(ProvenanceSchemaError, match=re.escape(message)):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda contract: contract.update({"invented": True}), "missing or unknown fields"),
        (lambda contract: contract.update({"role": "invented"}), "role must be a known string value"),
        (lambda contract: contract.update({"gradient_enabled": "false"}), "gradient_enabled must be a boolean"),
        (lambda contract: contract.update({"module_training": "false"}), "module_training must be a boolean"),
        (lambda contract: contract.update({"logical_step": True}), "logical_step must be an integer"),
        (lambda contract: contract.update({"episode_id": []}), "episode_id must be a non-empty string"),
        (lambda contract: contract["input_schema"].update({"keys": {}}), "input_schema.keys must be an array"),
        (
            lambda contract: contract.update({"batch_dimensions": ["time"]}),
            "batch dimensions must agree",
        ),
    ],
)
def test_workflow_provenance_strictly_decodes_contract_projection(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = _run().to_dict()
    mutation(payload["interaction_contract"])

    with pytest.raises(ProvenanceSchemaError, match=message):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
def test_workflow_provenance_validates_schema_key_projections() -> None:
    payload = _run().to_dict()
    key = payload["interaction_contract"]["input_schema"]["keys"][0]
    key["feature_shape"] = ["dynamic"]
    with pytest.raises(ProvenanceSchemaError, match="array of integers"):
        WorkflowProvenance.from_dict(payload)

    payload = _run().to_dict()
    key = payload["interaction_contract"]["input_schema"]["keys"][0]
    key["spec_constraints"] = {"callback": object()}
    with pytest.raises(ProvenanceSchemaError, match="JSON-compatible"):
        WorkflowProvenance.from_dict(payload)

    payload = _run().to_dict()
    key = payload["interaction_contract"]["input_schema"]["keys"][0]
    key["spec_constraints"] = {"low": float("-inf")}
    with pytest.raises(ProvenanceSchemaError, match="JSON-compatible"):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
def test_workflow_provenance_decodes_recurrent_and_multi_agent_contract_evidence() -> None:
    payload = _run().to_dict()
    contract = payload["interaction_contract"]
    contract["agent_dimension"] = "agent"
    contract["input_schema"]["keys"].extend(
        [
            {
                "path": ["state"],
                "role": "state",
                "presence": "required",
                "feature_shape": None,
                "spec_type": None,
                "spec_constraints": None,
            },
            {
                "path": ["is_init"],
                "role": "state",
                "presence": "required",
                "feature_shape": None,
                "spec_type": None,
                "spec_constraints": None,
            },
        ]
    )
    contract["output_schema"]["keys"].append(
        {
            "path": ["next", "state"],
            "role": "state",
            "presence": "produced",
            "feature_shape": None,
            "spec_type": None,
            "spec_constraints": None,
        }
    )
    contract["recurrent"] = {
        "transitions": [{"input_key": ["state"], "output_key": ["next", "state"]}],
        "reset_keys": [["is_init"]],
        "sequence_dimension": None,
        "burn_in": 0,
        "truncated_window": None,
        "collector_mode": "direct",
    }
    contract["multi_agent"] = {
        "topology": "parameter_shared",
        "group": "agents",
        "n_agents": 2,
        "target": {"role": "actor", "selector": {"group": "agents", "agents": [0, "blue"]}},
    }

    restored = WorkflowProvenance.from_dict(payload)

    assert restored.interaction_contract["recurrent"]["collector_mode"] == "direct"
    assert restored.interaction_contract["multi_agent"]["n_agents"] == 2


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda contract: contract.update({"gradient_enabled": True, "inference_mode": True}), "cannot both"),
        (lambda contract: contract.update({"autocast_enabled": True}), "requires autocast_device_type"),
        (
            lambda contract: contract.update(
                {
                    "multi_agent": {
                        "topology": "parameter_shared",
                        "group": "agents",
                        "n_agents": 2,
                        "target": {
                            "role": "actor",
                            "selector": {"group": "other", "agents": []},
                        },
                    }
                }
            ),
            "semantic target group must match",
        ),
    ],
)
def test_workflow_provenance_reapplies_canonical_contract_invariants(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = _run().to_dict()
    payload["interaction_contract"]["agent_dimension"] = "agent"
    mutation(payload["interaction_contract"])

    with pytest.raises(ProvenanceSchemaError, match=message):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("event_mutation", "message"),
    [
        (lambda events: events[1].update({"order": 3}), "order must be contiguous"),
        (lambda events: events[0].update({"kind": "failure"}), "ordered before/after pairs"),
        (lambda events: events[1].update({"phase": "loss"}), "match the interaction phase"),
        (lambda events: events[1].update({"key_shapes": {"action": [-1]}}), "integer array or null"),
        (lambda events: events[1].update({"error": "failed"}), "ordered before/after pairs"),
    ],
)
def test_workflow_provenance_rejects_invalid_lifecycle_evidence(
    event_mutation: Callable[[list[dict[str, object]]], None], message: str
) -> None:
    payload = _run().to_dict()
    event_mutation(payload["lifecycle_events"])

    with pytest.raises(ProvenanceSchemaError, match=message):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
def test_workflow_provenance_rejects_invalid_json_and_missing_dependencies() -> None:
    with pytest.raises(ProvenanceSchemaError, match="invalid provenance JSON"):
        WorkflowProvenance.from_json("{")
    with pytest.raises(ProvenanceSchemaError, match="must contain an object"):
        WorkflowProvenance.from_json("[]")

    payload = _run().to_dict()
    del payload["dependencies"]["tdhook"]
    with pytest.raises(ProvenanceSchemaError, match="dependency provenance is missing"):
        WorkflowProvenance.from_dict(payload)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan.update({"executions": {}}), "executions and compatibility must be arrays"),
        (
            lambda plan: plan["executions"][0].update({"model_passes": -1}),
            "model_passes must be a non-negative integer",
        ),
        (
            lambda plan: plan["executions"][0].update({"gradient_mode": ""}),
            "gradient_mode must be a non-empty string",
        ),
        (
            lambda plan: plan["executions"][0].update({"coexecuted": 1}),
            "coexecuted must be a boolean",
        ),
        (
            lambda plan: plan["executions"][0].update({"in_keys": ["observation"]}),
            r"in_keys\[0\] must be a non-empty array",
        ),
    ],
)
def test_workflow_provenance_rejects_malformed_plan_evidence(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = _run().to_dict()
    mutation(payload["workflow_plan"])

    with pytest.raises(ProvenanceSchemaError, match=message):
        WorkflowProvenance.from_dict(payload)
