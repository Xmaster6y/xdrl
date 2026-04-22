from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
from torchrl.record.loggers import get_logger


def _to_python(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def make_experiment_logger(cfg: DictConfig):
    backend = cfg.logger.backend
    if backend in (None, "", "stdout"):
        return None

    logger = get_logger(
        logger_type=backend,
        logger_name=cfg.logger.log_dir,
        experiment_name=cfg.logger.experiment_name,
        wandb_kwargs=_to_python(cfg.logger.get("wandb_kwargs", {})) or {},
        trackio_kwargs=_to_python(cfg.logger.get("trackio_kwargs", {})) or {},
    )
    if bool(cfg.logger.get("log_hparams", True)):
        logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
    return logger


def close_experiment_logger(cfg: DictConfig) -> None:
    if cfg.logger.backend != "wandb":
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()
