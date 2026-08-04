"""Serializable provenance for one xdrl interaction and TDHook method."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from xdrl.compatibility import installed_dependency_versions
from xdrl.interactions import InteractionDescriptor

PROVENANCE_SCHEMA_REVISION = 1


class ProvenanceSchemaError(ValueError):
    """A provenance payload is missing or incompatible with the current schema."""


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Reproduction metadata for an observed or intervened model invocation."""

    schema_revision: int
    model_id: str
    checkpoint_id: str | None
    interaction: Mapping[str, Any]
    selected_keys: tuple[tuple[str, ...], ...]
    target_paths: Mapping[str, str]
    exploration_mode: str | None
    gradient_enabled: bool
    batch_dimensions: tuple[str, ...]
    tdhook_method: Mapping[str, Any]
    dependencies: Mapping[str, str]
    code_revision: str

    def __post_init__(self) -> None:
        if self.schema_revision != PROVENANCE_SCHEMA_REVISION:
            raise ProvenanceSchemaError(
                f"unsupported provenance schema revision {self.schema_revision}; expected {PROVENANCE_SCHEMA_REVISION}"
            )
        if not self.model_id:
            raise ProvenanceSchemaError("model_id must be non-empty")
        if not self.code_revision:
            raise ProvenanceSchemaError("code_revision must be non-empty")
        missing = {"python", "torch", "tensordict", "torchrl", "tdhook", "xdrl"} - set(self.dependencies)
        if missing:
            raise ProvenanceSchemaError(f"dependency provenance is missing: {', '.join(sorted(missing))}")
        try:
            json.dumps(self.to_dict(), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ProvenanceSchemaError(f"manifest contains a non-serialisable value: {error}") from error

    @classmethod
    def capture(
        cls,
        descriptor: InteractionDescriptor,
        *,
        selected_keys: tuple[tuple[str, ...], ...],
        target_paths: Mapping[str, str],
        tdhook_method: Mapping[str, Any],
        code_revision: str,
        dependencies: Mapping[str, str] | None = None,
    ) -> ProvenanceManifest:
        """Build a manifest from the durable portion of a live interaction."""
        return cls(
            schema_revision=PROVENANCE_SCHEMA_REVISION,
            model_id=descriptor.model_id or descriptor.module_path,
            checkpoint_id=descriptor.checkpoint_id,
            interaction=json.loads(json.dumps(descriptor.to_dict())),
            selected_keys=selected_keys,
            target_paths=dict(target_paths),
            exploration_mode=descriptor.exploration_mode,
            gradient_enabled=descriptor.gradient_enabled,
            batch_dimensions=descriptor.batch_dimensions,
            tdhook_method=dict(tdhook_method),
            dependencies=dict(installed_dependency_versions() if dependencies is None else dependencies),
            code_revision=code_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible payload."""
        return json.loads(json.dumps(asdict(self)))

    def to_json(self) -> str:
        """Encode the manifest deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvenanceManifest:
        """Decode a manifest, rejecting unknown revisions and malformed fields."""
        revision = payload.get("schema_revision")
        if revision != PROVENANCE_SCHEMA_REVISION:
            raise ProvenanceSchemaError(
                f"unsupported provenance schema revision {revision!r}; expected {PROVENANCE_SCHEMA_REVISION}"
            )
        required = {field.name for field in fields(cls)}
        missing = required - set(payload)
        unknown = set(payload) - required
        if missing or unknown:
            detail = []
            if missing:
                detail.append(f"missing fields: {', '.join(sorted(missing))}")
            if unknown:
                detail.append(f"unknown fields: {', '.join(sorted(unknown))}")
            raise ProvenanceSchemaError("; ".join(detail))
        try:
            return cls(
                schema_revision=int(payload["schema_revision"]),
                model_id=str(payload["model_id"]),
                checkpoint_id=None if payload["checkpoint_id"] is None else str(payload["checkpoint_id"]),
                interaction=dict(payload["interaction"]),
                selected_keys=tuple(tuple(str(part) for part in key) for key in payload["selected_keys"]),
                target_paths={str(key): str(value) for key, value in payload["target_paths"].items()},
                exploration_mode=(
                    None if payload["exploration_mode"] is None else str(payload["exploration_mode"])
                ),
                gradient_enabled=bool(payload["gradient_enabled"]),
                batch_dimensions=tuple(str(value) for value in payload["batch_dimensions"]),
                tdhook_method=dict(payload["tdhook_method"]),
                dependencies={str(key): str(value) for key, value in payload["dependencies"].items()},
                code_revision=str(payload["code_revision"]),
            )
        except (TypeError, ValueError) as error:
            raise ProvenanceSchemaError(f"malformed provenance manifest: {error}") from error

    @classmethod
    def from_json(cls, payload: str) -> ProvenanceManifest:
        """Decode a manifest from JSON."""
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProvenanceSchemaError(f"invalid provenance JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ProvenanceSchemaError("provenance JSON must contain an object")
        return cls.from_dict(value)
