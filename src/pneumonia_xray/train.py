import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader

from pneumonia_xray.datamodule import ChestXrayDataModule
from pneumonia_xray.lightning_module import PneumoniaClassifier

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
    return "unknown"


def flatten_config(cfg: DictConfig, parent_key: str = "") -> dict:
    """Flatten nested config to dot-notation keys."""
    items = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, DictConfig):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value
    return items


def train_model(cfg: DictConfig) -> Path:
    """Train the pneumonia classifier using Hydra config.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Path to the best checkpoint.
    """
    # Set seed for reproducibility
    pl.seed_everything(cfg.trainer.seed, workers=True)

    git_commit = get_git_commit()
    logger.info(f"Git commit: {git_commit}")

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        run_name=cfg.logging.run_name,
        log_model=cfg.logging.log_model,
    )

    run_id = mlflow_logger.run_id
    logger.info(f"MLflow run ID: {run_id}")

    # Create directories
    checkpoint_dir = PROJECT_ROOT / cfg.trainer.checkpoint.dirpath
    run_plots_dir = PROJECT_ROOT / cfg.plots_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifacts will be saved to: {run_plots_dir}")

    # Log hyperparameters
    flat_config = flatten_config(cfg)
    flat_config["git_commit"] = git_commit
    mlflow_logger.log_hyperparams(flat_config)

    # Save resolved config to run folder
    config_path = run_plots_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved config to: {config_path}")

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

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=cfg.trainer.deterministic,
        logger=mlflow_logger,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test
    trainer.test(model, datamodule, ckpt_path="best")

    best_checkpoint_path = Path(checkpoint_callback.best_model_path)
    logger.info(f"Best checkpoint: {best_checkpoint_path}")

    # Generate plots and log artifacts
    try:
        best_model = PneumoniaClassifier.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        _generate_plots(best_model, datamodule.test_dataloader(), run_plots_dir)

        mlflow.log_artifacts(str(run_plots_dir), artifact_path="plots")
        logger.info(f"Logged artifacts from {run_plots_dir} to MLflow")

    except Exception as e:
        logger.error(f"Error generating plots or logging artifacts: {e}")

    return best_checkpoint_path


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

    logger.info(f"Plots saved to {plots_dir}")
