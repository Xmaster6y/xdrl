from pathlib import Path
import tomllib

import pytest
import torch
from tensordict.nn import TensorDictModule

from xdrl.compatibility import (
    PRIVATE_UPSTREAM_APIS,
    SUPPORTED_GIT_REVISIONS,
    installed_dependency_revisions,
    validate_runtime_compatibility,
)
from xdrl.tdhook import _reject_unsupported_module


@pytest.mark.upstream_compatibility
def test_private_upstream_inventory_is_owned_by_this_suite() -> None:
    owner = "tests/upstream_compatibility/test_private_apis.py"
    assert PRIVATE_UPSTREAM_APIS
    assert all(usage.owner_test == owner for usage in PRIVATE_UPSTREAM_APIS)
    assert all(usage.source_paths for usage in PRIVATE_UPSTREAM_APIS)
    assert all(usage.rationale for usage in PRIVATE_UPSTREAM_APIS)


@pytest.mark.upstream_compatibility
def test_compiled_module_marker_remains_fail_closed() -> None:
    policy = TensorDictModule(torch.nn.Linear(2, 1), in_keys=["observation"], out_keys=["action"])
    policy.module._orig_mod = policy.module

    with pytest.raises(NotImplementedError, match="torch.compile"):
        _reject_unsupported_module(policy)


@pytest.mark.upstream_compatibility
def test_supported_runtime_matches_the_lockfile_matrix() -> None:
    versions = validate_runtime_compatibility()
    revisions = installed_dependency_revisions()
    lock = tomllib.loads(Path("uv.lock").read_text())
    locked_sources = {
        package["name"]: package["source"]["git"] for package in lock["package"] if "git" in package.get("source", {})
    }

    assert set(versions) == {"python", "torch", "tensordict", "torchrl", "tdhook", "xdrl"}
    assert revisions == {requirement.distribution: requirement.commit for requirement in SUPPORTED_GIT_REVISIONS}
    for requirement in SUPPORTED_GIT_REVISIONS:
        assert locked_sources[requirement.distribution].endswith(f"#{requirement.commit}")
