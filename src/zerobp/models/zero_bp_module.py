"""Lightning model for the ZeroBP scaffold.

This file originally contained a tiny two layer MLP so that the training loop
could be exercised without implementing any of the architecture described in the
paper.  For the unit tests and for local experimentation it is convenient to
have a model that can operate on both vector inputs (used by the dummy
datamodule in the smoke tests) **and** on image like tensors.  The real ZeroBP
model is obviously far more sophisticated, however the implementation below
captures a few of its high level properties:

* a feature extractor that can either be a stack of fully–connected layers or a
  small convolutional network when images are provided;
* a head predicting arbitrary targets (e.g. correspondences, poses, …);
* a standard mean–squared error loss used in the tests.

The design keeps the interface identical to the previous stub so existing
configs and tests continue to function.  When images of shape `(B, C, H, W)` are
fed to :class:`ZeroBPModule` a lightweight convolutional backbone is used.  For
vector inputs of shape `(B, D)` it falls back to an MLP.  This allows the module
to be reused for prototyping while remaining faithful to the project structure
described in :mod:`README.md`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import lightning.pytorch as pl
import torch
from torch import nn


class MLP(nn.Module):
    """Simple multilayer perceptron used for vector inputs.

    Parameters
    ----------
    input_dim: int
        Size of the input feature vector.
    hidden_dims: List[int]
        Sizes of the hidden layers.
    output_dim: int
        Size of the output vector.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.net(x)


class ConvBackbone(nn.Module):
    """Very small convolutional feature extractor.

    The backbone is intentionally tiny – two convolutional blocks followed by
    global average pooling – so that it runs quickly during unit tests while
    still demonstrating the typical structure of the model used in the paper.
    """

    def __init__(self, in_channels: int, channels: Sequence[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev_c = in_channels
        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(prev_c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_c = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(prev_c, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        features = self.conv(x)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)


class ZeroBPModule(pl.LightningModule):
    """Minimal Lightning module used throughout the scaffold.

    Parameters
    ----------
    input_dim: int
        Dimensionality of vector inputs.  Ignored when ``cnn_channels`` is
        provided and images are used instead.
    output_dim: int
        Size of the prediction vector.
    hidden_dims: list[int] | None
        Hidden dimensions for the MLP branch.
    lr: float
        Learning rate for the optimizer.
    weight_decay: float
        L2 regularisation for the optimizer.
    cnn_channels: list[int] | None
        When not ``None`` the model expects image inputs with ``cnn_channels``
        specifying the number of feature maps for each convolutional block.
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        hidden_dims: List[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cnn_channels: List[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.save_hyperparameters()

        # Depending on the input type instantiate either an MLP or a tiny CNN
        if cnn_channels is None:
            self.model: nn.Module = MLP(input_dim, hidden_dims, output_dim)
        else:
            # For images we assume three channels by default.  This keeps the
            # interface simple for the smoke tests where random tensors are
            # passed in.
            self.model = ConvBackbone(in_channels=3, channels=cnn_channels, output_dim=output_dim)

        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Forward and training helpers
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        """Forward pass through the underlying model."""

        return self.model(x)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # ``batch_size`` ensures proper averaging by Lightning when using
        # different dataloaders or gradient accumulation.
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    # Lightning hooks --------------------------------------------------
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):  # pragma: no cover - exercised in tests
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

