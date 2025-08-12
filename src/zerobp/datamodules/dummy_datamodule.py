from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class SyntheticLinearDataset(Dataset):
    def __init__(
        self,
        n: int,
        input_dim: int,
        output_dim: int,
        noise_std: float = 0.05,
        W: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, input_dim, generator=g)
        self.W = W if W is not None else torch.randn(input_dim, output_dim, generator=g)
        self.b = b if b is not None else torch.randn(output_dim, generator=g)
        noise = noise_std * torch.randn(n, output_dim, generator=g)
        self.Y = self.X @ self.W + self.b + noise

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class DummyDataModule(pl.LightningDataModule):
    """
    A simple synthetic datamodule to validate training/evaluation loops.

    This serves as a placeholder until the intern implements real datasets as in the ZeroBP paper.
    """

    def __init__(
        self,
        n_train: int = 512,
        n_val: int = 128,
        n_test: int = 128,
        input_dim: int = 16,
        output_dim: int = 8,
        noise_std: float = 0.05,
        shuffle: bool = True,
        pin_memory: bool = True,
        dataloader: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.dl_cfg = dataloader or {"batch_size": 32, "num_workers": 0}

        # Placeholders set in setup()
        self.W: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Create a shared linear mapping for all splits
        g = torch.Generator().manual_seed(1234)
        self.W = torch.randn(self.input_dim, self.output_dim, generator=g)
        self.b = torch.randn(self.output_dim, generator=g)

        if stage in (None, "fit"):
            self._train = SyntheticLinearDataset(
                n=self.n_train,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                noise_std=self.noise_std,
                W=self.W,
                b=self.b,
                seed=1,
            )
            self._val = SyntheticLinearDataset(
                n=self.n_val,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                noise_std=self.noise_std,
                W=self.W,
                b=self.b,
                seed=2,
            )
        if stage in (None, "test"):
            self._test = SyntheticLinearDataset(
                n=self.n_test,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                noise_std=self.noise_std,
                W=self.W,
                b=self.b,
                seed=3,
            )

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=int(self.dl_cfg.get("batch_size", 32)),
            num_workers=int(self.dl_cfg.get("num_workers", 0)),
            shuffle=bool(self.shuffle),
            pin_memory=bool(self.pin_memory),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=int(self.dl_cfg.get("batch_size", 32)),
            num_workers=int(self.dl_cfg.get("num_workers", 0)),
            shuffle=False,
            pin_memory=bool(self.pin_memory),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=int(self.dl_cfg.get("batch_size", 32)),
            num_workers=int(self.dl_cfg.get("num_workers", 0)),
            shuffle=False,
            pin_memory=bool(self.pin_memory),
        )
