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

from typing import Dict, List, Sequence, Tuple

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


def _add_xy_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalised ``x``/``y`` coordinate channels to ``x``.

    Parameters
    ----------
    x:
        Tensor of shape ``(B, C, H, W)``.
    """

    b, _, h, w = x.shape
    device = x.device
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).expand(b, -1, -1, -1)
    return torch.cat([x, coords], dim=1)


class ConvBackbone(nn.Module):
    """Tiny convolutional encoder used for image inputs.

    Only a handful of convolutional blocks are used so that unit tests run
    quickly.  The module merely returns feature maps – prediction heads are
    implemented in :class:`ZeroBPModule`.
    """

    def __init__(self, in_channels: int, channels: Sequence[int], add_coords: bool = True) -> None:
        super().__init__()
        self.add_coords = add_coords

        prev_c = in_channels + (2 if add_coords else 0)
        layers: List[nn.Module] = []
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        if self.add_coords:
            x = _add_xy_channels(x)
        return self.conv(x)


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

        # Depending on the input type instantiate either an MLP or a CNN based
        # architecture that predicts dense correspondences as well as a global
        # pose vector.
        if cnn_channels is None:
            self.model: nn.Module = MLP(input_dim, hidden_dims, output_dim)
            self.image_model = None
        else:
            self.model = None  # type: ignore[assignment]
            self.backbone = ConvBackbone(in_channels=3, channels=cnn_channels, add_coords=True)
            c_out = cnn_channels[-1]
            self.coord_head = nn.Conv2d(c_out, 2, kernel_size=1)
            self.pose_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c_out, output_dim),
            )
            self.image_model = True

        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Forward and training helpers
    def forward(self, x: torch.Tensor):  # pragma: no cover - thin wrapper
        """Forward pass through the underlying model.

        When initialised for vector inputs an ``(N, output_dim)`` tensor is
        returned.  When using the CNN backbone a dictionary is produced with the
        dense correspondence map (``coords``) and the global pose vector
        (``pose``).
        """

        if self.image_model:
            feats = self.backbone(x)
            return {
                "coords": self.coord_head(feats),
                "pose": self.pose_head(feats),
            }
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        if isinstance(y_hat, dict):
            if not isinstance(y, dict):
                raise TypeError("Target must be a dict when model outputs a dict")
            losses = []
            for key, pred in y_hat.items():
                if key not in y:
                    continue
                tgt = y[key]
                l = self.criterion(pred, tgt)
                self.log(
                    f"{stage}/{key}_loss",
                    l,
                    prog_bar=(key == "pose"),
                    on_step=False,
                    on_epoch=True,
                    batch_size=x.size(0),
                )
                losses.append(l)
            loss = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=x.device)
        else:
            loss = self.criterion(y_hat, y)
            self.log(
                f"{stage}/loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=x.size(0),
            )
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

