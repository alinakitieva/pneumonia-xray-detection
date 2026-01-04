import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from pneumonia_xray.lightning_module import PneumoniaClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_transform(cfg: DictConfig) -> transforms.Compose:
    """Get inference transform matching training preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(cfg.data.normalize.mean),
                std=list(cfg.data.normalize.std),
            ),
        ]
    )


def predict_image(
    model: PneumoniaClassifier,
    image_path: Path,
    transform: transforms.Compose,
    threshold: float,
    labels: dict,
) -> dict:
    """Run prediction on a single image.

    Args:
        model: Loaded model.
        image_path: Path to image file.
        transform: Preprocessing transform.
        threshold: Classification threshold.
        labels: Label mapping dict.

    Returns:
        Dict with path, probability, and prediction.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Get device from model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    # Apply threshold
    pred_class = 1 if prob >= threshold else 0
    pred_label = labels[pred_class]

    return {
        "path": str(image_path),
        "probability": prob,
        "prediction": pred_label,
    }


def run_inference(cfg: DictConfig) -> list[dict]:
    """Run inference using Hydra config.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of prediction results.
    """
    # Validate input path
    if cfg.input.path is None:
        logger.error("No input path provided. Use: input.path=/path/to/image.png")
        return []

    input_path = Path(cfg.input.path)
    if not input_path.exists():
        # Try relative to project root
        input_path = PROJECT_ROOT / cfg.input.path
        if not input_path.exists():
            logger.error(f"Input path not found: {cfg.input.path}")
            return []

    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / cfg.checkpoint_path
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run training first: python -m pneumonia_xray.commands train")
        return []

    logger.info(f"Loading model from: {checkpoint_path}")
    model = PneumoniaClassifier.load_from_checkpoint(str(checkpoint_path))
    model.train(False)

    # Get transform and config
    transform = get_transform(cfg)
    threshold = cfg.postprocess.threshold
    labels = {int(k): v for k, v in cfg.postprocess.labels.items()}

    # Collect image paths
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Folder: find all images
        image_paths = (
            list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.jpeg"))
        )
        if not image_paths:
            logger.error(f"No images found in: {input_path}")
            return []

    logger.info(f"Running inference on {len(image_paths)} image(s)")

    # Run predictions
    results = []
    for img_path in image_paths:
        try:
            result = predict_image(model, img_path, transform, threshold, labels)
            results.append(result)
            print(
                f"path={result['path']} prob={result['probability']:.4f} "
                f"pred={result['prediction']}"
            )
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    return results
