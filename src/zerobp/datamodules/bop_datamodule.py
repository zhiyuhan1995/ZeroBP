from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from PIL import Image
from torchvision import transforms as T


def _maybe_load_json(path: str | Path) -> Optional[Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _get_entry(container: Any, idx: int) -> Optional[Any]:
    """Access by index for list-like or by key for dict-like JSON structures used in BOP.
    Returns None if entry not present.
    """
    if container is None:
        return None
    if isinstance(container, list):
        if 0 <= idx < len(container):
            return container[idx]
        return None
    # Some BOP tools produce dicts keyed by string of frame id
    key = str(idx)
    return container.get(key)


class BOPImageDataset(Dataset):
    """Minimal reader for BOP-format datasets.

    Expects structure like:
      <bop_root>/<dataset>/<split>/<scene_id>/
        - rgb/*.png
        - scene_camera.json
        - scene_gt.json

    This is intended as a scaffold: adapt/extend for specific benchmarks and
    to parse depth, masks, objects, etc.
    """

    def __init__(
        self,
        bop_root: str,
        dataset: str,
        split: str = "train_pbr",
        img_size: Optional[Sequence[int]] = None,
        normalize: bool = True,
        max_images: Optional[int] = None,
        return_tuple_for_stub: bool = False,
    ) -> None:
        super().__init__()
        self.bop_root = bop_root
        self.dataset = dataset
        self.split = split
        self.img_size = tuple(img_size) if img_size is not None else None
        self.normalize = normalize
        self.max_images = max_images
        self.return_tuple_for_stub = return_tuple_for_stub

        self.base = os.path.join(bop_root, dataset, split)
        if not os.path.isdir(self.base):
            raise FileNotFoundError(f"BOP split directory not found: {self.base}")

        # Build index of images and associated scene JSON handles
        self.samples: List[Tuple[int, int, str]] = []  # (scene_id, im_id, rgb_path)
        self._scene_camera: Dict[int, Any] = {}
        self._scene_gt: Dict[int, Any] = {}

        scene_dirs = sorted([p for p in glob(os.path.join(self.base, "*")) if os.path.isdir(p)])
        for sd in scene_dirs:
            scene_id = int(os.path.basename(sd)) if os.path.basename(sd).isdigit() else -1
            cam = _maybe_load_json(os.path.join(sd, "scene_camera.json"))
            gt = _maybe_load_json(os.path.join(sd, "scene_gt.json"))
            self._scene_camera[scene_id] = cam
            self._scene_gt[scene_id] = gt

            rgb_dir = os.path.join(sd, "rgb")
            if not os.path.isdir(rgb_dir):
                # Some datasets might use "color" instead of "rgb"
                rgb_dir = os.path.join(sd, "color")
            if not os.path.isdir(rgb_dir):
                # Skip scenes without images
                continue
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in sorted(glob(os.path.join(rgb_dir, ext))):
                    stem = Path(img_path).stem
                    try:
                        im_id = int(stem)
                    except ValueError:
                        # filenames might not be numeric; skip if so
                        continue
                    self.samples.append((scene_id, im_id, img_path))

        if self.max_images is not None:
            self.samples = self.samples[: int(self.max_images)]

        # Build transforms
        tfs: List[Any] = []
        if self.img_size is not None:
            tfs.append(T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR))
        tfs.append(T.ToTensor())
        if self.normalize:
            tfs.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.tf = T.Compose(tfs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        scene_id, im_id, img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        image_t = self.tf(img)

        cam = self._scene_camera.get(scene_id)
        gt = self._scene_gt.get(scene_id)
        cam_entry = _get_entry(cam, im_id)
        gt_entry = _get_entry(gt, im_id)

        K = None
        if cam_entry is not None and "cam_K" in cam_entry:
            K_list = cam_entry["cam_K"]
            try:
                K = torch.tensor(K_list, dtype=torch.float32).view(3, 3)
            except Exception:
                K = None
        meta = {
            "scene_id": scene_id,
            "im_id": im_id,
            "path": img_path,
            "K": K,
            "gt": gt_entry if gt_entry is not None else [],
        }

        if self.return_tuple_for_stub:
            # Provide a dummy regression target to keep the current stub model runnable if needed.
            # Here we just predict the mean pixel value per channel as a simple target placeholder.
            target = image_t.mean(dim=(1, 2))  # shape [3]
            return image_t.view(-1), target  # flatten image as features (very large!)

        return {"image": image_t, "meta": meta}


class BOPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        bop_root: str = os.path.join("data", "bop"),
        dataset: str = "ycbv",
        train_split: str = "train_pbr",
        val_split: str = "val",
        test_split: str = "test",
        img_size: Optional[Sequence[int]] = None,
        normalize: bool = True,
        max_images: Optional[int] = None,
        return_tuple_for_stub: bool = False,
        pin_memory: bool = True,
        dataloader: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.bop_root = bop_root
        self.dataset = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.img_size = img_size
        self.normalize = normalize
        self.max_images = max_images
        self.return_tuple_for_stub = return_tuple_for_stub
        self.pin_memory = pin_memory
        self.dl_cfg = dataloader or {"batch_size": 2, "num_workers": 4}

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train = BOPImageDataset(
                bop_root=self.bop_root,
                dataset=self.dataset,
                split=self.train_split,
                img_size=self.img_size,
                normalize=self.normalize,
                max_images=self.max_images,
                return_tuple_for_stub=self.return_tuple_for_stub,
            )
            self._val = BOPImageDataset(
                bop_root=self.bop_root,
                dataset=self.dataset,
                split=self.val_split,
                img_size=self.img_size,
                normalize=self.normalize,
                max_images=self.max_images,
                return_tuple_for_stub=self.return_tuple_for_stub,
            )
        if stage in (None, "test"):
            self._test = BOPImageDataset(
                bop_root=self.bop_root,
                dataset=self.dataset,
                split=self.test_split,
                img_size=self.img_size,
                normalize=self.normalize,
                max_images=self.max_images,
                return_tuple_for_stub=self.return_tuple_for_stub,
            )

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=int(self.dl_cfg.get("batch_size", 2)),
            num_workers=int(self.dl_cfg.get("num_workers", 4)),
            shuffle=True,
            pin_memory=bool(self.pin_memory),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=int(self.dl_cfg.get("batch_size", 2)),
            num_workers=int(self.dl_cfg.get("num_workers", 4)),
            shuffle=False,
            pin_memory=bool(self.pin_memory),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=int(self.dl_cfg.get("batch_size", 2)),
            num_workers=int(self.dl_cfg.get("num_workers", 4)),
            shuffle=False,
            pin_memory=bool(self.pin_memory),
        )
