import pytest

import xdrl.compatibility as compatibility
from xdrl.compatibility import (
    SUPPORT_DEFINITIONS,
    CompatibilityBoundaryError,
    ConformanceCheck,
    SupportLevel,
    VersionRequirement,
    WORKFLOW_CONFORMANCE,
    _validate_version,
)


def test_support_levels_have_test_backed_definitions() -> None:
    assert set(SUPPORT_DEFINITIONS) == set(SupportLevel)
    assert "full declared conformance suite" in SUPPORT_DEFINITIONS[SupportLevel.SUPPORTED]
    assert "not the full supported matrix" in SUPPORT_DEFINITIONS[SupportLevel.EXPERIMENTAL]
    assert "rejected explicitly" in SUPPORT_DEFINITIONS[SupportLevel.UNSUPPORTED]


def test_every_advertised_workflow_boundary_names_a_complete_conformance_suite() -> None:
    assert {suite.boundary for suite in WORKFLOW_CONFORMANCE} == {"TDHookWorkflowRunner"}
    suite = WORKFLOW_CONFORMANCE[0]
    assert suite.support is SupportLevel.SUPPORTED
    assert set(suite.checks) == set(ConformanceCheck)
    assert suite.test_path.endswith("test_tdhook_workflow.py")


def test_version_failures_name_the_failed_boundary() -> None:
    with pytest.raises(CompatibilityBoundaryError, match="boundary 'version:example'.*tested requirement") as error:
        _validate_version(VersionRequirement("example", ">=2"), "1.0")

    assert error.value.boundary == "version:example"


def test_installed_versions_skip_non_applicable_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    requirements = (
        VersionRequirement("present", "==1"),
        VersionRequirement("missing", "==1", marker="python_version < '0'"),
    )
    requested: list[str] = []
    monkeypatch.setattr(compatibility, "SUPPORTED_DEPENDENCIES", requirements)
    monkeypatch.setattr(
        compatibility,
        "_installed_version",
        lambda distribution: requested.append(distribution) or "1.0",
    )

    versions = compatibility.installed_dependency_versions()

    assert requested == ["present"]
    assert versions["present"] == "1.0"
    assert "missing" not in versions
