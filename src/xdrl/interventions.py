"""Explicit, checked interventions for typed RL interactions.

The public objects here describe *mechanics*, not causal conclusions. XDRL
owns only TensorDict boundary edits applied by ``RuntimeInteractionContext``.
TDHook owns activation, gradient, and parameter interventions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDictBase
from xdrl.types import KeyPresence, TensorDictKey, TensorDictSchema

if TYPE_CHECKING:
    from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext


class InterventionTarget(str, Enum):
    """The execution boundary an intervention edits."""

    TENSORDICT = "tensordict"


class InterventionTiming(str, Enum):
    """Whether the value is edited before or after the selected boundary."""

    INPUT = "input"
    OUTPUT = "output"


class InterventionValidationError(ValueError):
    """An intervention cannot safely be applied to an interaction."""


TensorTransform = Callable[[torch.Tensor], torch.Tensor]
InteractionPredicate = Callable[["InteractionContract"], bool]


@dataclass(frozen=True, slots=True)
class InterventionScope:
    """Optional semantic constraints selecting supported interaction runs."""

    phases: frozenset["InteractionPhase"] | None = None
    roles: frozenset[Any] | None = None
    environment: str | None = None
    time_dimension: str | None = None
    agent_dimension: str | None = None
    exploration_mode: str | None = None
    predicate: InteractionPredicate | None = field(default=None, compare=False, repr=False)

    def matches(self, contract: "InteractionContract") -> bool:
        return (
            (self.phases is None or contract.phase in self.phases)
            and (self.roles is None or contract.role in self.roles)
            and (self.environment is None or contract.environment == self.environment)
            and (self.time_dimension is None or contract.time_dimension == self.time_dimension)
            and (self.agent_dimension is None or contract.agent_dimension == self.agent_dimension)
            and (self.exploration_mode is None or contract.exploration_mode == self.exploration_mode)
            and (self.predicate is None or self.predicate(contract))
        )


@dataclass(frozen=True, slots=True)
class Intervention:
    """One replacement or transform at an explicit, checked target.

    ``key`` identifies a declared TensorDict input or output. Exactly one of
    ``replacement`` and ``transform`` must be supplied. Model-internal targets
    belong to TDHook's ``Target`` and ``HookSession`` APIs.
    """

    identifier: str
    target: InterventionTarget
    timing: InterventionTiming
    replacement: torch.Tensor | None = field(default=None, repr=False, compare=False)
    transform: TensorTransform | None = field(default=None, repr=False, compare=False)
    key: TensorDictKey | None = None
    module_path: str | None = None
    scope: InterventionScope = field(default_factory=InterventionScope)

    def __post_init__(self) -> None:
        if not self.identifier:
            raise InterventionValidationError("intervention identifier must be non-empty")
        if (self.replacement is None) == (self.transform is None):
            raise InterventionValidationError("provide exactly one of replacement or transform")
        if self.key is None or self.module_path is not None:
            raise InterventionValidationError("TensorDict interventions require key and forbid module_path")

    def applies_to(self, contract: "InteractionContract") -> bool:
        return self.scope.matches(contract)

    def edit_tensor(self, value: torch.Tensor, *, label: str) -> torch.Tensor:
        result = self.replacement if self.replacement is not None else self.transform(value)  # type: ignore[misc]
        if not isinstance(result, torch.Tensor):
            raise InterventionValidationError(f"{self.identifier}: {label} transform must return a torch.Tensor")
        if result.shape != value.shape or result.dtype != value.dtype or result.device != value.device:
            raise InterventionValidationError(
                f"{self.identifier}: {label} replacement must preserve shape={tuple(value.shape)!r}, "
                f"dtype={value.dtype}, and device={value.device}; got shape={tuple(result.shape)!r}, "
                f"dtype={result.dtype}, device={result.device}"
            )
        return result


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """Tensor-free provenance for an intervention application."""

    identifier: str
    target: InterventionTarget
    timing: InterventionTiming
    interaction_id: str
    checkpoint_id: str | None
    order: int


@dataclass(frozen=True, slots=True)
class PairedInterventionResult:
    """Outputs from matched control and intervention executions."""

    interaction_id: str
    checkpoint_id: str | None
    baseline: TensorDictBase
    intervention: TensorDictBase


class InterventionController:
    """Apply TensorDict edits and retain compact intervention provenance."""

    def __init__(self, interventions: Iterable[Intervention] = ()) -> None:
        self.interventions = tuple(interventions)
        self.records: list[InterventionRecord] = []

    def validate(self, interaction: "RuntimeInteractionContext") -> None:
        for intervention in self.interventions:
            if not intervention.applies_to(interaction.contract):
                continue
            if intervention.target is not InterventionTarget.TENSORDICT:
                continue
            schema = (
                interaction.input_schema
                if intervention.timing is InterventionTiming.INPUT
                else interaction.output_schema
            )
            _validate_key(intervention, schema)

    def apply(
        self, interaction: "RuntimeInteractionContext", tensordict: TensorDictBase, timing: InterventionTiming
    ) -> TensorDictBase:
        schema = interaction.input_schema if timing is InterventionTiming.INPUT else interaction.output_schema
        for intervention in self.interventions:
            if intervention.target is not InterventionTarget.TENSORDICT or intervention.timing is not timing:
                continue
            if not intervention.applies_to(interaction.contract):
                continue
            assert intervention.key is not None
            value = tensordict.get(intervention.key)
            if not isinstance(value, torch.Tensor):
                raise InterventionValidationError(
                    f"{intervention.identifier}: key {_display_key(intervention.key)} is not a tensor"
                )
            tensordict.set(intervention.key, intervention.edit_tensor(value, label=_display_key(intervention.key)))
            # Revalidate immediately: this includes feature shape, batch semantics,
            # device/dtype membership, and any TensorSpec constraint.
            if timing is InterventionTiming.INPUT:
                schema.validate_inputs(tensordict)
            else:
                schema.validate_outputs(tensordict)
            self.records.append(
                InterventionRecord(
                    intervention.identifier,
                    intervention.target,
                    timing,
                    interaction.contract.identity,
                    interaction.contract.checkpoint_id,
                    len(self.records),
                )
            )
        return tensordict


def run_paired(
    baseline: "RuntimeInteractionContext",
    intervention: "RuntimeInteractionContext",
    tensordict: TensorDictBase,
) -> PairedInterventionResult:
    """Run matched baseline and intervention contexts over independent inputs.

    The caller explicitly constructs both contexts, which keeps model ownership
    and execution modes in TorchRL.  This helper only enforces the provenance
    identity needed to compare their outputs safely.
    """
    baseline_contract = baseline.contract
    intervention_contract = intervention.contract
    if (
        baseline_contract.identity != intervention_contract.identity
        or baseline_contract.checkpoint_id != intervention_contract.checkpoint_id
    ):
        raise InterventionValidationError(
            "paired executions must share interaction identity and checkpoint provenance"
        )
    initial_rng_state = _rng_state()
    post_baseline_rng_state = initial_rng_state
    try:
        with baseline:
            baseline_output = baseline.invoke(tensordict.clone())
        post_baseline_rng_state = _rng_state()
        _set_rng_state(initial_rng_state)
        with intervention:
            intervention_output = intervention.invoke(tensordict.clone())
    finally:
        # A pair consumes the same random stream as one baseline invocation.
        # This both matches stochastic executions and avoids leaking an extra
        # random draw into the caller's next interaction.
        _set_rng_state(post_baseline_rng_state)
    return PairedInterventionResult(
        baseline_contract.identity,
        baseline_contract.checkpoint_id,
        baseline_output,
        intervention_output,
    )


def _validate_key(intervention: Intervention, schema: TensorDictSchema) -> None:
    assert intervention.key is not None
    expected = KeyPresence.REQUIRED if intervention.timing is InterventionTiming.INPUT else KeyPresence.PRODUCED
    if not any(entry.key == intervention.key and entry.presence is expected for entry in schema.keys):
        raise InterventionValidationError(
            f"{intervention.identifier}: key {_display_key(intervention.key)} is not a declared {expected.value} key"
        )


def _display_key(key: TensorDictKey) -> str:
    return "/".join(key) if isinstance(key, tuple) else key


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Snapshot CPU and, when present, every CUDA generator state."""
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def _set_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
