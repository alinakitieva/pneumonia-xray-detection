from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader

from pneumonia_xray.datamodule import ChestXrayDataModule
from pneumonia_xray.lightning_module import PneumoniaClassifier

PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"


def train_model(
    batch_size: int = 32,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    max_epochs: int = 10,
    seed: int = 42,
    pos_weight: float = 2.9,
) -> Path:
    """Train the pneumonia classifier.

    Args:
        batch_size: Batch size for training.
        num_workers: Number of workers for data loading.
        learning_rate: Learning rate for optimizer.
        max_epochs: Maximum number of epochs.
        seed: Random seed for reproducibility.
        pos_weight: Positive class weight for loss function.

    Returns:
        Path to the best checkpoint.
    """
    # Set seed for reproducibility
    pl.seed_everything(seed, workers=True)

    # Create directories
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize data module
    datamodule = ChestXrayDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Initialize model
    model = PneumoniaClassifier(
        learning_rate=learning_rate,
        pos_weight=pos_weight,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=ARTIFACTS_DIR,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test
    trainer.test(model, datamodule, ckpt_path="best")

    # Generate plots
    best_model = PneumoniaClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    _generate_plots(best_model, datamodule.test_dataloader())

    return Path(checkpoint_callback.best_model_path)


def _generate_plots(model: PneumoniaClassifier, test_loader: DataLoader) -> None:
    """Generate confusion matrix and ROC curve plots."""
    model.train(False)
    device = next(model.parameters()).device

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).int()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_labels,
        all_preds,
        display_labels=["NORMAL", "PNEUMONIA"],
        ax=ax,
        cmap="Blues",
    )
    ax.set_title("Confusion Matrix")
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        all_labels,
        all_probs,
        ax=ax,
        name="Pneumonia Classifier",
    )
    ax.set_title("ROC Curve")
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {PLOTS_DIR}")
