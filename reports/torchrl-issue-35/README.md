# TorchRL compatibility reproductions for issue #35

These scripts preserve the five focused observations recovered from the
retired `news` design spike.  They are evidence for upstream TorchRL reports,
not XDRL conformance tests and not requests for a local workaround.

## Pinned environment

Run each script from this repository's root after `uv sync --locked`:

```console
uv run python reports/torchrl-issue-35/sac_loss_config.py
```

The commands target the exact lockfile revision:

* TorchRL `0.12.0+g5b2bc08b`
  (`5b2bc08b034bf228bfa8563629980b939d59b089`);
* TensorDict `0.12.2`;
* PyTorch `2.11.0`.

Each script asserts the desired upstream behaviour.  A failing assertion is
therefore the reproduction to attach to an upstream report; a successful exit
means that particular observation is no longer applicable at the pinned
revision.  Do not add these to XDRL's supported conformance suite.

## Report ledger

| Script | Observation | Result at pinned revision | Upstream disposition |
| --- | --- | --- | --- |
| `sac_loss_config.py` | continuous `SACLossConfig` forwards discrete-only fields | reproduced: all three fields reach continuous `SACLoss` | file upstream |
| `transformed_env_warning.py` | auto-unwrap warning names obsolete `v0.9` target | reproduced: emitted warning contains `0.9` | file upstream |
| `vector_reward_autoreset.py` | vector Gymnasium autoreset mixes reward shapes | reproduced: vector construction raises an incompatible reward-spec shape error | file upstream |
| `log_validation_reward.py` | validation hook closes reusable evaluation environment | reproduced: second hook call raises `ClosedEnvironmentError` | file upstream |
| `gymnasium_pixel_imports.py` | Gymnasium pixel detection imports legacy Gym/CV2 | reproduced: both modules appear in `sys.modules` | file upstream |

When filing or updating an upstream issue, copy the exact command, complete
traceback or success output, the lockfile revision above, and only the
corresponding script.  Record the resulting upstream URL (or the reason the
reproduction is retired) in this table.  This directory deliberately carries
no trainer, configuration, logging, checkpoint, or environment workaround.
