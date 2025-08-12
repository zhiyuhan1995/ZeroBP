from typing import Any, List

import hydra
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from hydra.utils import instantiate


def _to_container(cfg_section: Any):
    return OmegaConf.to_container(cfg_section, resolve=True)


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    # Reproducibility
    if "seed" in cfg and cfg.seed is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    # Instantiate model and datamodule
    if not hasattr(cfg, "model") or "_target_" not in cfg.model:
        raise ValueError("Config 'model' must specify a _target_. See configs/model/zero_bp_stub.yaml")
    model = instantiate(cfg.model)

    if not hasattr(cfg, "datamodule") or "_target_" not in cfg.datamodule:
        raise ValueError("Config 'datamodule' must specify a _target_. See configs/datamodule/dummy.yaml")
    datamodule = instantiate(cfg.datamodule)

    # Loggers
    loggers: List[Any] = []
    if hasattr(cfg, "logger") and cfg.logger is not None:
        if "_target_" in cfg.logger:
            loggers = [instantiate(cfg.logger)]
        elif isinstance(cfg.logger, list):
            loggers = [instantiate(lg) for lg in cfg.logger]

    # Callbacks
    callbacks = []
    if hasattr(cfg, "callbacks") and cfg.callbacks is not None:
        if isinstance(cfg.callbacks, list):
            callbacks = [instantiate(cb) for cb in cfg.callbacks]
        elif "_target_" in cfg.callbacks:
            callbacks = [instantiate(cfg.callbacks)]

    # Trainer
    trainer_kwargs = {}
    if hasattr(cfg, "trainer") and cfg.trainer is not None:
        trainer_kwargs = _to_container(cfg.trainer)

    # Fallback CSV logger if none provided
    if len(loggers) == 0:
        from lightning.pytorch.loggers import CSVLogger
        loggers = [CSVLogger(save_dir="logs", name="csv")]

    trainer: Trainer = Trainer(logger=loggers, callbacks=callbacks, **trainer_kwargs)

    # Evaluate
    stage = str(cfg.get("stage", "test"))
    ckpt_path = cfg.get("ckpt_path", None)

    if stage == "validate" or stage == "val":
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
