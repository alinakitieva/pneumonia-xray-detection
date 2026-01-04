from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
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
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None
        self.test_dataset: Optional[ImageFolder] = None

    @property
    def train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @property
    def val_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
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
