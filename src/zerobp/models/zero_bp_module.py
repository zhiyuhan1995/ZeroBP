from typing import List, Tuple

import torch
from torch import nn
import lightning.pytorch as pl


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZeroBPModule(pl.LightningModule):
    """
    Placeholder LightningModule for ZeroBP.
    Replace the simple MLP with the architecture and losses described in the paper.
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        hidden_dims: List[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.save_hyperparameters()

        self.model = MLP(input_dim, hidden_dims, output_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
