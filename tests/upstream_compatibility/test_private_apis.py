from pathlib import Path

import pytest
import torch
from tensordict.nn import TensorDictModule
from torchrl.trainers.algorithms.configs.common import _normalize_hydra_key
from torchrl.trainers.algorithms.configs.trainers import _make_dqn_trainer, _make_ppo_trainer
from torchrl.trainers.trainers import Trainer, _resolve_module

from xdrl.compatibility import PRIVATE_UPSTREAM_APIS, validate_runtime_compatibility
from xdrl.tdhook import _reject_unsupported_module


@pytest.mark.upstream_compatibility
def test_private_upstream_inventory_is_owned_by_this_suite() -> None:
    owner = "tests/upstream_compatibility/test_private_apis.py"
    assert PRIVATE_UPSTREAM_APIS
    assert all(usage.owner_test == owner for usage in PRIVATE_UPSTREAM_APIS)
    assert all(usage.source_paths for usage in PRIVATE_UPSTREAM_APIS)
    assert all(usage.rationale for usage in PRIVATE_UPSTREAM_APIS)


@pytest.mark.upstream_compatibility
def test_torchrl_private_surfaces_remain_available() -> None:
    assert callable(_normalize_hydra_key)
    assert callable(_resolve_module)
    assert callable(Trainer._log)
    assert callable(_make_ppo_trainer)
    assert callable(_make_dqn_trainer)


@pytest.mark.upstream_compatibility
def test_compiled_module_marker_remains_fail_closed() -> None:
    policy = TensorDictModule(torch.nn.Linear(2, 1), in_keys=["observation"], out_keys=["action"])
    policy.module._orig_mod = policy.module

    with pytest.raises(NotImplementedError, match="torch.compile"):
        _reject_unsupported_module(policy)


@pytest.mark.upstream_compatibility
def test_supported_runtime_matches_the_lockfile_matrix() -> None:
    versions = validate_runtime_compatibility()

    assert set(versions) == {"python", "torch", "tensordict", "torchrl", "tdhook", "xdrl"}
    assert Path("uv.lock").is_file()
