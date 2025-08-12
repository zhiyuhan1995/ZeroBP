import os
import sys
import pytest


@pytest.mark.parametrize("fast_dev_run", [True])
def test_training_smoke(fast_dev_run):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch import Trainer
        from zerobp.models import ZeroBPModule
        from zerobp.datamodules import DummyDataModule
    except Exception as e:
        pytest.skip(f"Dependencies not installed for smoke test: {e}")

    dm = DummyDataModule(n_train=64, n_val=16, n_test=16, input_dim=8, output_dim=4, dataloader={"batch_size": 4, "num_workers": 0})
    model = ZeroBPModule(input_dim=8, output_dim=4, hidden_dims=[16], lr=1e-3)

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        max_epochs=1,
        log_every_n_steps=1,
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=dm)
