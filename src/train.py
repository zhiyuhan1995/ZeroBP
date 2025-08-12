import os
from typing import Any, List

import hydra
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from hydra.utils import instantiate


def _to_container(cfg_section: Any):
    return OmegaConf.to_container(cfg_section, resolve=True)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Reproducibility
    if "seed" in cfg and cfg.seed is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    # Instantiate model
    if not hasattr(cfg, "model") or "_target_" not in cfg.model:
        raise ValueError("Config 'model' must specify a _target_. See configs/model/zero_bp_stub.yaml")
    model = instantiate(cfg.model)

    # Instantiate datamodule
    if not hasattr(cfg, "datamodule") or "_target_" not in cfg.datamodule:
        raise ValueError("Config 'datamodule' must specify a _target_. See configs/datamodule/dummy.yaml")
    datamodule = instantiate(cfg.datamodule)

    # Instantiate loggers (single or list)
    loggers: List[Any] = []
    if hasattr(cfg, "logger") and cfg.logger is not None:
        if "_target_" in cfg.logger:
            loggers = [instantiate(cfg.logger)]
        elif isinstance(cfg.logger, list):
            loggers = [instantiate(lg) for lg in cfg.logger]

    # Instantiate callbacks (list or single)
    callbacks = []
    if hasattr(cfg, "callbacks") and cfg.callbacks is not None:
        if isinstance(cfg.callbacks, list):
            callbacks = [instantiate(cb) for cb in cfg.callbacks]
        elif "_target_" in cfg.callbacks:
            callbacks = [instantiate(cfg.callbacks)]

    # Trainer configuration
    trainer_kwargs = {}
    if hasattr(cfg, "trainer") and cfg.trainer is not None:
        trainer_kwargs = _to_container(cfg.trainer)  # plain dict

    # Ensure default logger save_dir
    if len(loggers) == 0:
        # Lazy import to avoid optional dependency if not used
        from lightning.pytorch.loggers import CSVLogger
        loggers = [CSVLogger(save_dir="logs", name="csv")]

    trainer: Trainer = Trainer(logger=loggers, callbacks=callbacks, **trainer_kwargs)

    # Optionally resume
    ckpt_path = cfg.get("ckpt_path", None)

    # Fit
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Optionally test after training if configured
    if getattr(cfg, "test_after_fit", False):
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
