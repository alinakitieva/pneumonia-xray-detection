from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader

from pneumonia_xray.datamodule import ChestXrayDataModule
from pneumonia_xray.lightning_module import PneumoniaClassifier

PROJECT_ROOT = Path(__file__).parent.parent.parent


def train_model(cfg: DictConfig) -> Path:
    """Train the pneumonia classifier using Hydra config.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Path to the best checkpoint.
    """
    # Set seed for reproducibility
    pl.seed_everything(cfg.trainer.seed, workers=True)

    # Create directories
    checkpoint_dir = PROJECT_ROOT / cfg.trainer.checkpoint.dirpath
    plots_dir = PROJECT_ROOT / cfg.plots_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data module from config
    datamodule = ChestXrayDataModule.from_config(cfg)

    # Initialize model from config
    model = PneumoniaClassifier(
        learning_rate=cfg.trainer.learning_rate,
        pos_weight=cfg.model.pos_weight,
    )

    # Callbacks from config
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.trainer.checkpoint.filename,
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.mode,
        save_top_k=cfg.trainer.checkpoint.save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.trainer.early_stopping.monitor,
        patience=cfg.trainer.early_stopping.patience,
        mode=cfg.trainer.early_stopping.mode,
    )

    # Trainer from config
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=cfg.trainer.deterministic,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test
    trainer.test(model, datamodule, ckpt_path="best")

    # Generate plots
    best_model = PneumoniaClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    _generate_plots(best_model, datamodule.test_dataloader(), plots_dir)

    return Path(checkpoint_callback.best_model_path)


def _generate_plots(
    model: PneumoniaClassifier, test_loader: DataLoader, plots_dir: Path
) -> None:
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
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(plots_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {plots_dir}")
