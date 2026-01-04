from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from pneumonia_xray.data import get_data_path


class ChestXrayDataModule(pl.LightningDataModule):
    """DataModule for Chest X-ray Pneumonia dataset."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        normalize_mean: Optional[list[float]] = None,
        normalize_std: Optional[list[float]] = None,
        augmentation_flip: bool = True,
        augmentation_rotation: int = 10,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Normalization (ImageNet defaults)
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

        # Augmentation
        self.augmentation_flip = augmentation_flip
        self.augmentation_rotation = augmentation_rotation

        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None
        self.test_dataset: Optional[ImageFolder] = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ChestXrayDataModule":
        """Create DataModule from Hydra config."""
        data_cfg = cfg.data
        return cls(
            data_dir=Path(data_cfg.root) if data_cfg.root else None,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            image_size=data_cfg.image_size,
            normalize_mean=list(data_cfg.normalize.mean),
            normalize_std=list(data_cfg.normalize.std),
            augmentation_flip=data_cfg.augmentation.horizontal_flip,
            augmentation_rotation=data_cfg.augmentation.rotation_degrees,
        )

    @property
    def train_transform(self) -> transforms.Compose:
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if self.augmentation_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if self.augmentation_rotation > 0:
            transform_list.append(transforms.RandomRotation(self.augmentation_rotation))
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )
        return transforms.Compose(transform_list)

    @property
    def val_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )

    def prepare_data(self) -> None:
        """Download data if needed."""
        get_data_path()

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage."""
        if self.data_dir is None:
            self.data_dir = get_data_path()

        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                root=self.data_dir / "train",
                transform=self.train_transform,
            )
            self.val_dataset = ImageFolder(
                root=self.data_dir / "val",
                transform=self.val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                root=self.data_dir / "test",
                transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
