import logging
import shutil
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_transform(cfg: DictConfig) -> transforms.Compose:
    """Get preprocessing transform matching training."""
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


def prepare_model_repository(cfg: DictConfig) -> Path:
    """Prepare Triton model repository with ONNX model.

    Args:
        cfg: Hydra configuration.

    Returns:
        Path to model repository.
    """
    model_repo = PROJECT_ROOT / cfg.triton.model_repository
    model_dir = model_repo / cfg.triton.model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy ONNX model
    onnx_src = PROJECT_ROOT / cfg.triton.onnx_path
    onnx_dst = model_dir / "model.onnx"

    if not onnx_src.exists():
        logger.error(f"ONNX model not found: {onnx_src}")
        logger.error("Run export first: python -m pneumonia_xray.commands export")
        raise FileNotFoundError(f"ONNX model not found: {onnx_src}")

    # Check for external data file
    onnx_data_src = onnx_src.with_suffix(".onnx.data")

    logger.info(f"Copying ONNX model to: {onnx_dst}")
    shutil.copy2(onnx_src, onnx_dst)

    if onnx_data_src.exists():
        onnx_data_dst = model_dir / "model.onnx.data"
        logger.info(f"Copying ONNX data to: {onnx_data_dst}")
        shutil.copy2(onnx_data_src, onnx_data_dst)

    # Copy metadata for reference
    metadata_src = PROJECT_ROOT / "artifacts/export/metadata.json"
    if metadata_src.exists():
        metadata_dst = model_repo / cfg.triton.model_name / "metadata.json"
        shutil.copy2(metadata_src, metadata_dst)
        logger.info(f"Copied metadata to: {metadata_dst}")

    logger.info(f"Model repository prepared at: {model_repo}")
    return model_repo


def run_triton_inference(cfg: DictConfig) -> list[dict]:
    """Run inference using Triton Inference Server.

    Args:
        cfg: Hydra configuration.

    Returns:
        List of prediction results.
    """
    try:
        import tritonclient.http as httpclient
    except ImportError:
        logger.error(
            "tritonclient not installed. Install with: pip install tritonclient[http]"
        )
        return []

    # Validate input path
    if cfg.input.path is None:
        logger.error("No input path provided. Use: input.path=/path/to/image.png")
        return []

    input_path = Path(cfg.input.path)
    if not input_path.exists():
        input_path = PROJECT_ROOT / cfg.input.path
        if not input_path.exists():
            logger.error(f"Input path not found: {cfg.input.path}")
            return []

    # Connect to Triton server
    triton_url = f"{cfg.triton.host}:{cfg.triton.http_port}"
    logger.info(f"Connecting to Triton server at: {triton_url}")

    try:
        client = httpclient.InferenceServerClient(url=triton_url)
        if not client.is_server_live():
            logger.error("Triton server is not live")
            return []
        if not client.is_model_ready(cfg.triton.model_name):
            logger.error(f"Model '{cfg.triton.model_name}' is not ready")
            return []
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {e}")
        return []

    # Get transform and config
    transform = get_transform(cfg)
    threshold = cfg.postprocess.threshold
    labels = {int(k): v for k, v in cfg.postprocess.labels.items()}

    # Collect image paths
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = (
            list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.jpeg"))
        )
        if not image_paths:
            logger.error(f"No images found in: {input_path}")
            return []

    logger.info(f"Running Triton inference on {len(image_paths)} image(s)")

    # Run predictions
    results = []
    for img_path in image_paths:
        try:
            result = predict_single_image(
                client,
                cfg.triton.model_name,
                img_path,
                transform,
                threshold,
                labels,
            )
            results.append(result)
            print(
                f"path={result['path']} prob={result['probability']:.4f} "
                f"pred={result['prediction']}"
            )
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    return results


def predict_single_image(
    client,
    model_name: str,
    image_path: Path,
    transform: transforms.Compose,
    threshold: float,
    labels: dict,
) -> dict:
    """Run prediction on a single image via Triton.

    Args:
        client: Triton HTTP client.
        model_name: Name of the model in Triton.
        image_path: Path to image file.
        transform: Preprocessing transform.
        threshold: Classification threshold.
        labels: Label mapping dict.

    Returns:
        Dict with path, probability, and prediction.
    """
    import tritonclient.http as httpclient

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

    # Create Triton input
    inputs = [
        httpclient.InferInput("input", input_tensor.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(input_tensor)

    # Create Triton output
    outputs = [
        httpclient.InferRequestedOutput("logits"),
    ]

    # Run inference
    response = client.infer(model_name, inputs, outputs=outputs)
    logit = response.as_numpy("logits")[0]

    # Apply sigmoid and threshold
    prob = 1 / (1 + np.exp(-logit))
    pred_class = 1 if prob >= threshold else 0
    pred_label = labels[pred_class]

    return {
        "path": str(image_path),
        "probability": float(prob),
        "prediction": pred_label,
    }


def run_prepare_serving(cfg: DictConfig) -> dict:
    """Prepare model repository for Triton serving.

    Args:
        cfg: Hydra configuration.

    Returns:
        Dict with paths to prepared files.
    """
    results = {
        "model_repository": None,
        "model_path": None,
        "config_path": None,
    }

    try:
        model_repo = prepare_model_repository(cfg)
        results["model_repository"] = str(model_repo)
        results["model_path"] = str(
            model_repo / cfg.triton.model_name / "1" / "model.onnx"
        )
        results["config_path"] = str(
            model_repo / cfg.triton.model_name / "config.pbtxt"
        )
    except Exception as e:
        logger.error(f"Failed to prepare model repository: {e}")

    return results
