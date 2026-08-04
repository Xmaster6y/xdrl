import pytest

from xdrl.compatibility import (
    ADAPTER_CONFORMANCE,
    SUPPORT_DEFINITIONS,
    CompatibilityBoundaryError,
    ConformanceCheck,
    SupportLevel,
    VersionRequirement,
    _validate_version,
)


def test_support_levels_have_test_backed_definitions() -> None:
    assert set(SUPPORT_DEFINITIONS) == set(SupportLevel)
    assert "full declared conformance suite" in SUPPORT_DEFINITIONS[SupportLevel.SUPPORTED]
    assert "not the full supported matrix" in SUPPORT_DEFINITIONS[SupportLevel.EXPERIMENTAL]
    assert "rejected explicitly" in SUPPORT_DEFINITIONS[SupportLevel.UNSUPPORTED]


def test_every_advertised_adapter_names_a_complete_conformance_suite() -> None:
    assert {suite.adapter for suite in ADAPTER_CONFORMANCE} == {"TDHookInteractionAdapter"}
    suite = ADAPTER_CONFORMANCE[0]
    assert suite.support is SupportLevel.SUPPORTED
    assert set(suite.checks) == set(ConformanceCheck)
    assert suite.test_path.endswith("test_tdhook_adapter.py")


def test_version_failures_name_the_failed_boundary() -> None:
    with pytest.raises(CompatibilityBoundaryError, match="boundary 'version:example'.*tested requirement") as error:
        _validate_version(VersionRequirement("example", ">=2"), "1.0")

    assert error.value.boundary == "version:example"
