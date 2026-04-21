"""Bootstrap SARL DQN training entrypoint.

Examples:

```bash
uv run -m scripts.sarl.gymnasium_dqn
uv run -m scripts.sarl.gymnasium_dqn env.id=Acrobot-v1
```
"""

from __future__ import annotations

import hydra
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig


def make_env(cfg: DictConfig) -> None:
    pylogger.info("Preparing SARL env '{}' with {} envs", cfg.env.id, cfg.env.num_envs)


def make_modules(cfg: DictConfig) -> None:
    pylogger.info("Preparing SARL DQN modules with hidden_dim={} depth={}", cfg.model.hidden_dim, cfg.model.depth)


def make_trainer(cfg: DictConfig) -> None:
    pylogger.info(
        "Preparing SARL DQN trainer frames_per_batch={} total_frames={}",
        cfg.collector.frames_per_batch,
        cfg.collector.total_frames,
    )


@hydra.main(config_path="../../configs/sarl", config_name="gymnasium_dqn", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    pylogger.info("Starting SARL DQN bootstrap for env {}", cfg.env.id)

    make_env(cfg)
    make_modules(cfg)
    make_trainer(cfg)

    pylogger.info("SARL DQN bootstrap is ready for implementation")


if __name__ == "__main__":
    main()
