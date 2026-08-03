"""Explicit, checked interventions for typed RL interactions.

The public objects here describe *mechanics*, not causal conclusions.  A
TensorDict intervention is applied by ``RuntimeInteractionContext`` while
activation and gradient interventions are exposed as a TDHook factory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDictBase
from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookHandle, resolve_submodule_path
from tdhook.modules import HookedModule

from xdrl.types import KeyPresence, TensorDictKey, TensorDictSchema

if TYPE_CHECKING:
    from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext


class InterventionTarget(str, Enum):
    """The execution boundary an intervention edits."""

    TENSORDICT = "tensordict"
    ACTIVATION = "activation"
    GRADIENT = "gradient"


class InterventionTiming(str, Enum):
    """Whether the value is edited before or after the selected boundary."""

    INPUT = "input"
    OUTPUT = "output"


class InterventionValidationError(ValueError):
    """An intervention cannot safely be applied to an interaction."""


TensorTransform = Callable[[torch.Tensor], torch.Tensor]
InteractionPredicate = Callable[["InteractionDescriptor"], bool]


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

    def matches(self, descriptor: "InteractionDescriptor") -> bool:
        return (
            (self.phases is None or descriptor.phase in self.phases)
            and (self.roles is None or descriptor.role in self.roles)
            and (self.environment is None or descriptor.environment == self.environment)
            and (self.time_dimension is None or descriptor.time_dimension == self.time_dimension)
            and (self.agent_dimension is None or descriptor.agent_dimension == self.agent_dimension)
            and (self.exploration_mode is None or descriptor.exploration_mode == self.exploration_mode)
            and (self.predicate is None or self.predicate(descriptor))
        )


@dataclass(frozen=True, slots=True)
class Intervention:
    """One replacement or transform at an explicit, checked target.

    ``key`` is required for ``TENSORDICT`` interventions and ``module_path``
    is required for TDHook activation and gradient interventions.  Exactly one
    of ``replacement`` and ``transform`` must be supplied.
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
        if self.target is InterventionTarget.TENSORDICT:
            if self.key is None or self.module_path is not None:
                raise InterventionValidationError("TensorDict interventions require key and forbid module_path")
        elif self.module_path is None or self.key is not None:
            raise InterventionValidationError(f"{self.target.value} interventions require module_path and forbid key")

    def applies_to(self, descriptor: "InteractionDescriptor") -> bool:
        return self.scope.matches(descriptor)

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
            if not intervention.applies_to(interaction.descriptor):
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
            if not intervention.applies_to(interaction.descriptor):
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
                    interaction.descriptor.identity,
                    interaction.descriptor.checkpoint_id,
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
    baseline_descriptor = baseline.descriptor
    intervention_descriptor = intervention.descriptor
    if (
        baseline_descriptor.identity != intervention_descriptor.identity
        or baseline_descriptor.checkpoint_id != intervention_descriptor.checkpoint_id
    ):
        raise InterventionValidationError(
            "paired executions must share interaction identity and checkpoint provenance"
        )
    with baseline:
        baseline_output = baseline.invoke(tensordict.clone())
    with intervention:
        intervention_output = intervention.invoke(tensordict.clone())
    return PairedInterventionResult(
        baseline_descriptor.identity,
        baseline_descriptor.checkpoint_id,
        baseline_output,
        intervention_output,
    )


class TDHookInterventionFactory(HookingContextFactory):
    """Install activation/gradient edits through TDHook's managed lifecycle."""

    def __init__(self, interaction: "RuntimeInteractionContext", interventions: Sequence[Intervention]) -> None:
        super().__init__()
        self._interaction = interaction
        self._interventions = tuple(interventions)
        for intervention in self._interventions:
            if intervention.target is InterventionTarget.TENSORDICT:
                raise InterventionValidationError(
                    "TDHookInterventionFactory accepts activation or gradient interventions only"
                )
            if intervention.applies_to(interaction.descriptor):
                assert intervention.module_path is not None
                try:
                    resolve_submodule_path(interaction.module, intervention.module_path)
                except ValueError as error:
                    raise InterventionValidationError(
                        f"{intervention.identifier}: cannot resolve TDHook target {intervention.module_path!r}"
                    ) from error

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []
        for intervention in self._interventions:
            if not intervention.applies_to(self._interaction.descriptor):
                continue
            assert intervention.module_path is not None
            direction = _hook_direction(intervention)

            def callback(
                *, intervention: Intervention = intervention, direction: str = direction, **kwargs: Any
            ) -> torch.Tensor | tuple[torch.Tensor, ...]:
                container = (
                    kwargs["output"]
                    if direction == "fwd"
                    else (
                        kwargs["args"]
                        if direction == "fwd_pre"
                        else kwargs["grad_input"]
                        if direction == "bwd"
                        else kwargs["grad_output"]
                    )
                )
                value = container if isinstance(container, torch.Tensor) else container[-1]
                if not isinstance(value, torch.Tensor):
                    raise InterventionValidationError(
                        f"{intervention.identifier}: TDHook target {intervention.module_path!r} did not receive a tensor"
                    )
                edited = intervention.edit_tensor(value, label=f"TDHook target {intervention.module_path}")
                return edited if isinstance(container, torch.Tensor) else (*container[:-1], edited)

            handles.append(module.set(intervention.module_path, None, callback=callback, direction=direction))
        return MultiHookHandle(handles)


def _validate_key(intervention: Intervention, schema: TensorDictSchema) -> None:
    assert intervention.key is not None
    expected = KeyPresence.REQUIRED if intervention.timing is InterventionTiming.INPUT else KeyPresence.PRODUCED
    if not any(entry.key == intervention.key and entry.presence is expected for entry in schema.keys):
        raise InterventionValidationError(
            f"{intervention.identifier}: key {_display_key(intervention.key)} is not a declared {expected.value} key"
        )


def _hook_direction(intervention: Intervention) -> str:
    if intervention.target is InterventionTarget.ACTIVATION:
        return "fwd_pre" if intervention.timing is InterventionTiming.INPUT else "fwd"
    return "bwd" if intervention.timing is InterventionTiming.INPUT else "bwd_pre"


def _display_key(key: TensorDictKey) -> str:
    return "/".join(key) if isinstance(key, tuple) else key
