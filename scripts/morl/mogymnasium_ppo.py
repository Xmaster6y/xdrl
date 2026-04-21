"""Bootstrap MORL PPO training entrypoint.

Examples:

```bash
uv run -m scripts.morl.mogymnasium_ppo
uv run -m scripts.morl.mogymnasium_ppo env.id=mo-hopper-v4
```
"""

from __future__ import annotations

import hydra
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig


def make_env(cfg: DictConfig) -> None:
    pylogger.info("Preparing MORL env '{}' with {} envs", cfg.env.id, cfg.env.num_envs)


def make_modules(cfg: DictConfig) -> None:
    pylogger.info("Preparing MORL PPO modules with hidden_dim={} depth={}", cfg.model.hidden_dim, cfg.model.depth)


def make_trainer(cfg: DictConfig) -> None:
    pylogger.info(
        "Preparing MORL PPO trainer frames_per_batch={} total_frames={}",
        cfg.collector.frames_per_batch,
        cfg.collector.total_frames,
    )


@hydra.main(config_path="../../configs/morl", config_name="mogymnasium_ppo", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    pylogger.info("Starting MORL PPO bootstrap for env {}", cfg.env.id)

    make_env(cfg)
    make_modules(cfg)
    make_trainer(cfg)

    pylogger.info("MORL PPO bootstrap is ready for implementation")


if __name__ == "__main__":
    main()
