"""
Run a useless demo script.

Run with:

```bash
uv run -m scripts.demo
```
"""

from loguru import logger
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="demo.yaml", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"This is a demo script with an important parameter: {cfg.important_parameter}")


if __name__ == "__main__":
    main()
