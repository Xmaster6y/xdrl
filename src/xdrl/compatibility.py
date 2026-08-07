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
    VersionRequirement(
        "torch",
        "==2.13.*",
        marker="((python_full_version >= '3.12' and sys_platform != 'darwin') or (python_full_version == '3.12.*' and sys_platform == 'darwin')) or (python_full_version < '3.12' and sys_platform != 'darwin')",
    ),
    VersionRequirement(
        "torch",
        "==2.14.0.dev20260805",
        marker="(python_full_version >= '3.13' and sys_platform == 'darwin') or (python_full_version < '3.12' and sys_platform == 'darwin')",
    ),
    VersionRequirement("tensordict", "==0.13.0+g54a147b"),
    VersionRequirement("torchrl", "==0.13.0+gae421b98"),
    VersionRequirement("tdhook", "==0.2.0"),
    VersionRequirement("xdrl", "==0.2.0"),
)
SUPPORTED_GIT_REVISIONS = (
    GitRevisionRequirement("tensordict", "54a147b2d3c21ac407661a26e27a9b0c37e7fbd3"),
    GitRevisionRequirement("torchrl", "ae421b98d0dba86e5ab0b24917d1e64f376ee6f9"),
    GitRevisionRequirement("tdhook", "ea51fe37094d19ddbdacb5db685583a5d1b88b3a"),
)
# dependency-snapshot: end


class ConformanceCheck(str, Enum):
    """Observable workflow-runner properties requiring executable evidence."""

    SCHEMA_PRESERVATION = "schema_preservation"
    OUTPUT_PARITY = "output_parity"
    LIFECYCLE_CLEANUP = "lifecycle_cleanup"
    EXCEPTION_SAFETY = "exception_safety"
    LAZY_MATERIALISATION = "lazy_materialisation"
    PLAN_DELEGATION = "plan_delegation"
    MODEL_PASS_ACCOUNTING = "model_pass_accounting"
    PROVENANCE_ROUND_TRIP = "provenance_round_trip"


@dataclass(frozen=True, slots=True)
class ConformanceSuite:
    """Named test suite supporting one advertised integration boundary."""

    boundary: str
    test_path: str
    checks: tuple[ConformanceCheck, ...]
    support: SupportLevel


WORKFLOW_CONFORMANCE = (
    ConformanceSuite(
        boundary="TDHookWorkflowRunner",
        test_path="tests/behavioural_parity/test_tdhook_workflow.py",
        checks=(
            ConformanceCheck.SCHEMA_PRESERVATION,
            ConformanceCheck.OUTPUT_PARITY,
            ConformanceCheck.LIFECYCLE_CLEANUP,
            ConformanceCheck.EXCEPTION_SAFETY,
            ConformanceCheck.LAZY_MATERIALISATION,
            ConformanceCheck.PROVENANCE_ROUND_TRIP,
        ),
        support=SupportLevel.SUPPORTED,
    ),
    ConformanceSuite(
        boundary="TDHookWorkflowRunner",
        test_path="tests/integration/test_tdhook_workflow.py",
        checks=(
            ConformanceCheck.PLAN_DELEGATION,
            ConformanceCheck.MODEL_PASS_ACCOUNTING,
        ),
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
)


def installed_dependency_versions() -> dict[str, str]:
    """Return versions needed to reproduce the supported runtime boundary."""
    result = {
        requirement.distribution: _installed_version(requirement.distribution)
        for requirement in SUPPORTED_DEPENDENCIES
        if requirement.applies()
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
