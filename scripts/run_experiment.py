"""
Run a trainer based experiment.

Example usage:
```bash
uv run -m scripts.run_experiment --config-name vmas_qmix
```
"""

from __future__ import annotations

import hydra

from torchrl.trainers.algorithms.configs import *  # noqa: F401,F403

from xdrl.configs import register_configs

register_configs()


@hydra.main(config_path="../configs", config_name="vmas_ppo", version_base="1.1")
def main(cfg) -> None:
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
