"""Test-backed compatibility and conformance declarations for xdrl.

Installation is not treated as evidence of behavioural support.  This module
names the versions and suites which establish each advertised boundary.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from typing import Any

from packaging.markers import Marker
from packaging.specifiers import SpecifierSet
from packaging.version import Version


class SupportLevel(str, Enum):
    """Evidence level attached to a dependency or adapter boundary."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


SUPPORT_DEFINITIONS = {
    SupportLevel.SUPPORTED: "covered by the full declared conformance suite in required CI",
    SupportLevel.EXPERIMENTAL: "covered by unit and integration tests but not the full supported matrix",
    SupportLevel.UNSUPPORTED: "rejected explicitly or not covered by an advertised conformance suite",
}


class CompatibilityBoundaryError(RuntimeError):
    """A named compatibility boundary failed validation."""

    def __init__(self, boundary: str, detail: str) -> None:
        self.boundary = boundary
        self.detail = detail
        super().__init__(f"compatibility boundary {boundary!r} failed: {detail}")


@dataclass(frozen=True, slots=True)
class VersionRequirement:
    """One dependency range exercised by the supported CI matrix."""

    distribution: str
    specifier: str
    support: SupportLevel = SupportLevel.SUPPORTED
    marker: str | None = None

    def applies(self) -> bool:
        """Return whether this conditional requirement targets this runtime."""
        return self.marker is None or Marker(self.marker).evaluate()


@dataclass(frozen=True, slots=True)
class GitRevisionRequirement:
    """An immutable Git revision exercised by the supported CI matrix."""

    distribution: str
    commit: str


SUPPORTED_PYTHON = VersionRequirement("python", ">=3.11,<3.14")
# dependency-snapshot: start
SUPPORTED_DEPENDENCIES = (
    VersionRequirement("torch", "==2.11.*"),
    VersionRequirement("tensordict", "==0.12.2"),
    VersionRequirement("torchrl", "==0.12.0+g5b2bc08b"),
    VersionRequirement("tdhook", "==0.1.3"),
    VersionRequirement("xdrl", "==0.1.0"),
)
SUPPORTED_GIT_REVISIONS = (
    GitRevisionRequirement("torchrl", "5b2bc08b034bf228bfa8563629980b939d59b089"),
    GitRevisionRequirement("tdhook", "1a01cd3ea3bc04b9fe60877604d2116b610af108"),
)
# dependency-snapshot: end


class ConformanceCheck(str, Enum):
    """Observable adapter properties which require executable evidence."""

    SCHEMA_PRESERVATION = "schema_preservation"
    OUTPUT_PARITY = "output_parity"
    LIFECYCLE_CLEANUP = "lifecycle_cleanup"
    EXCEPTION_SAFETY = "exception_safety"
    LAZY_MATERIALISATION = "lazy_materialisation"
    TARGET_PATH_RESOLUTION = "target_path_resolution"


@dataclass(frozen=True, slots=True)
class ConformanceSuite:
    """Named test suite supporting one advertised adapter."""

    adapter: str
    test_path: str
    checks: tuple[ConformanceCheck, ...]
    support: SupportLevel


ADAPTER_CONFORMANCE = (
    ConformanceSuite(
        adapter="TDHookInteractionAdapter",
        test_path="tests/behavioural_parity/test_tdhook_adapter.py",
        checks=tuple(ConformanceCheck),
        support=SupportLevel.SUPPORTED,
    ),
)


@dataclass(frozen=True, slots=True)
class PrivateAPIUsage:
    """Owned reliance on a private upstream surface."""

    package: str
    symbol: str
    source_paths: tuple[str, ...]
    owner_test: str
    rationale: str


PRIVATE_UPSTREAM_APIS = (
    PrivateAPIUsage(
        "torch",
        "torch.nn.Module._orig_mod",
        ("src/xdrl/tdhook.py",),
        "tests/upstream_compatibility/test_private_apis.py",
        "detect compiled descendants before TDHook installation",
    ),
    PrivateAPIUsage(
        "torchrl",
        "torchrl.trainers.Trainer._log",
        ("src/xdrl/configs/hooks.py",),
        "tests/upstream_compatibility/test_private_apis.py",
        "emit pre-evaluation metrics through the trainer logger",
    ),
    PrivateAPIUsage(
        "torchrl",
        "torchrl.trainers.algorithms.configs.common._normalize_hydra_key",
        ("src/xdrl/configs/hooks.py",),
        "tests/upstream_compatibility/test_private_apis.py",
        "normalise Hydra-configured TensorDict keys identically to TorchRL",
    ),
    PrivateAPIUsage(
        "torchrl",
        "torchrl.trainers.trainers._resolve_module",
        ("src/xdrl/configs/hooks.py", "src/xdrl/trainer_hooks/checkpoints.py"),
        "tests/upstream_compatibility/test_private_apis.py",
        "resolve configured trainer module paths using TorchRL semantics",
    ),
    PrivateAPIUsage(
        "torchrl",
        "torchrl.record.loggers.wandb._step_registry",
        ("src/xdrl/trainer_hooks/logging.py",),
        "tests/upstream_compatibility/test_private_apis.py",
        "detect whether TorchRL's W&B logger has an uncommitted scalar row",
    ),
)


def installed_dependency_versions() -> dict[str, str]:
    """Return versions needed to reproduce the supported runtime boundary."""
    result = {
        requirement.distribution: _installed_version(requirement.distribution)
        for requirement in SUPPORTED_DEPENDENCIES
    }
    result["python"] = ".".join(str(part) for part in sys.version_info[:3])
    return result


def validate_runtime_compatibility() -> dict[str, str]:
    """Validate the current interpreter and dependencies against the tested matrix."""
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    _validate_version(SUPPORTED_PYTHON, python_version)
    versions = installed_dependency_versions()
    for requirement in SUPPORTED_DEPENDENCIES:
        if requirement.applies():
            _validate_version(requirement, versions[requirement.distribution])
    revisions = installed_dependency_revisions()
    for requirement in SUPPORTED_GIT_REVISIONS:
        if revisions[requirement.distribution] != requirement.commit:
            raise CompatibilityBoundaryError(
                f"revision:{requirement.distribution}",
                f"installed {revisions[requirement.distribution]!r}, tested revision is {requirement.commit!r}",
            )
    return versions


def installed_dependency_revisions() -> dict[str, str]:
    """Return immutable Git revisions for dependencies pinned from source."""
    return {
        requirement.distribution: _installed_git_revision(requirement.distribution)
        for requirement in SUPPORTED_GIT_REVISIONS
    }


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise CompatibilityBoundaryError(f"dependency:{distribution}", "distribution is not installed") from error


def _installed_git_revision(distribution_name: str) -> str:
    try:
        payload = distribution(distribution_name).read_text("direct_url.json")
    except PackageNotFoundError as error:
        raise CompatibilityBoundaryError(f"dependency:{distribution_name}", "distribution is not installed") from error
    if payload is None:
        raise CompatibilityBoundaryError(f"revision:{distribution_name}", "direct URL metadata is unavailable")
    try:
        direct_url: dict[str, Any] = json.loads(payload)
        commit = direct_url["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise CompatibilityBoundaryError(
            f"revision:{distribution_name}", "Git commit metadata is unavailable"
        ) from error
    if not isinstance(commit, str) or not commit:
        raise CompatibilityBoundaryError(f"revision:{distribution_name}", "Git commit metadata is invalid")
    return commit


def _validate_version(requirement: VersionRequirement, actual: str) -> None:
    if Version(actual) not in SpecifierSet(requirement.specifier):
        raise CompatibilityBoundaryError(
            f"version:{requirement.distribution}",
            f"installed {actual!r}, tested requirement is {requirement.specifier!r}",
        )
