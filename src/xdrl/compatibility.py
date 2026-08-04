"""Test-backed compatibility and conformance declarations for xdrl.

Installation is not treated as evidence of behavioural support.  This module
names the versions and suites which establish each advertised boundary.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version

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


SUPPORTED_PYTHON = VersionRequirement("python", ">=3.11,<3.14")
SUPPORTED_DEPENDENCIES = (
    VersionRequirement("torch", "==2.11.*"),
    VersionRequirement("tensordict", "==0.12.2"),
    VersionRequirement("torchrl", "==0.12.0+g5b2bc08b"),
    VersionRequirement("tdhook", "==0.1.3.dev0"),
    VersionRequirement("xdrl", "==0.1.0"),
)


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
        "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer",
        ("torchrl.trainers.algorithms.configs",),
        "tests/upstream_compatibility/test_private_apis.py",
        "Hydra target used by the advertised PPO configuration",
    ),
    PrivateAPIUsage(
        "torchrl",
        "torchrl.trainers.algorithms.configs.trainers._make_dqn_trainer",
        ("torchrl.trainers.algorithms.configs",),
        "tests/upstream_compatibility/test_private_apis.py",
        "Hydra target used by the advertised DQN and QMIX configurations",
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
        _validate_version(requirement, versions[requirement.distribution])
    return versions


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise CompatibilityBoundaryError(f"dependency:{distribution}", "distribution is not installed") from error


def _validate_version(requirement: VersionRequirement, actual: str) -> None:
    if Version(actual) not in SpecifierSet(requirement.specifier):
        raise CompatibilityBoundaryError(
            f"version:{requirement.distribution}",
            f"installed {actual!r}, tested requirement is {requirement.specifier!r}",
        )
