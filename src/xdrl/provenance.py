"""Serializable provenance for one xdrl interaction and TDHook method."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from packaging.version import InvalidVersion, Version

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
        if type(self.schema_revision) is not int or self.schema_revision != PROVENANCE_SCHEMA_REVISION:
            raise ProvenanceSchemaError(
                f"unsupported provenance schema revision {self.schema_revision}; expected {PROVENANCE_SCHEMA_REVISION}"
            )
        if not self.model_id:
            raise ProvenanceSchemaError("model_id must be non-empty")
        if not self.code_revision:
            raise ProvenanceSchemaError("code_revision must be non-empty")
        if not isinstance(self.dependencies, Mapping):
            raise ProvenanceSchemaError("dependencies must be a mapping")
        missing = _required_dependency_names() - set(self.dependencies)
        if missing:
            raise ProvenanceSchemaError(f"dependency provenance is missing: {', '.join(sorted(missing))}")
        _validate_dependencies(self.dependencies)
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
        if not isinstance(payload, Mapping):
            raise ProvenanceSchemaError("provenance manifest must contain an object")
        revision = payload.get("schema_revision")
        if type(revision) is not int or revision != PROVENANCE_SCHEMA_REVISION:
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
        interaction = _mapping(payload["interaction"], "interaction")
        target_paths = _string_mapping(payload["target_paths"], "target_paths")
        tdhook_method = _mapping(payload["tdhook_method"], "tdhook_method")
        dependencies = _validate_dependencies(_string_mapping(payload["dependencies"], "dependencies"))
        return cls(
            schema_revision=revision,
            model_id=_nonempty_string(payload["model_id"], "model_id"),
            checkpoint_id=_optional_string(payload["checkpoint_id"], "checkpoint_id"),
            interaction=dict(interaction),
            selected_keys=_key_paths(payload["selected_keys"], "selected_keys"),
            target_paths=target_paths,
            exploration_mode=_optional_string(payload["exploration_mode"], "exploration_mode"),
            gradient_enabled=_boolean(payload["gradient_enabled"], "gradient_enabled"),
            batch_dimensions=_strings(payload["batch_dimensions"], "batch_dimensions"),
            tdhook_method=dict(tdhook_method),
            dependencies=dependencies,
            code_revision=_nonempty_string(payload["code_revision"], "code_revision"),
        )

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


def _required_dependency_names() -> set[str]:
    return {"python", "torch", "tensordict", "torchrl", "tdhook", "xdrl"}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProvenanceSchemaError(f"{field} must be an object with string keys")
    return value


def _string_mapping(value: Any, field: str) -> dict[str, str]:
    mapping = _mapping(value, field)
    return {key: _nonempty_string(item, f"{field}.{key}") for key, item in mapping.items()}


def _validate_dependencies(value: Mapping[str, str]) -> dict[str, str]:
    missing = _required_dependency_names() - set(value)
    if missing:
        raise ProvenanceSchemaError(f"dependency provenance is missing: {', '.join(sorted(missing))}")
    dependencies = _string_mapping(value, "dependencies")
    for name, dependency_version in dependencies.items():
        try:
            Version(dependency_version)
        except InvalidVersion as error:
            raise ProvenanceSchemaError(f"dependencies.{name} must be a valid version") from error
    return dependencies


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceSchemaError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field)


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ProvenanceSchemaError(f"{field} must be a boolean")
    return value


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    return tuple(_nonempty_string(item, f"{field}[{index}]") for index, item in enumerate(value))


def _key_paths(value: Any, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    paths = []
    for index, path in enumerate(value):
        if not isinstance(path, list) or not path:
            raise ProvenanceSchemaError(f"{field}[{index}] must be a non-empty array")
        paths.append(tuple(_nonempty_string(part, f"{field}[{index}]") for part in path))
    return tuple(paths)
