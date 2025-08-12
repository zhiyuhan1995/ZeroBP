#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test for the scaffold
python src/train.py env=local experiment=debug datamodule.dataloader.batch_size=8 trainer.max_epochs=1 trainer.limit_train_batches=2
