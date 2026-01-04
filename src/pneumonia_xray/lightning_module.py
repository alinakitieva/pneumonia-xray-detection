from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision.models import ResNet18_Weights, resnet18


class PneumoniaClassifier(pl.LightningModule):
    """Lightning module for pneumonia classification."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load pretrained ResNet-18 and replace final layer
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

        # Loss function with class imbalance handling
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")

        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels = batch
        logits = self(images)
        labels_float = labels.float()
        loss = self.criterion(logits, labels_float)
        preds = torch.sigmoid(logits) > 0.5
        return loss, preds, labels

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, labels = self._shared_step(batch)
        self.train_accuracy(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
