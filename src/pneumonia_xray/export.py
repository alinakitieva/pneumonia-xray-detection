import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from pneumonia_xray.lightning_module import PneumoniaClassifier

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


def load_sample_input(
    cfg: DictConfig,
    sample_path: Path | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load or create a sample input tensor.

    Returns both PyTorch tensor and NumPy array for validation.
    """
    transform = get_transform(cfg)

    if sample_path and sample_path.exists():
        image = Image.open(sample_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
    else:
        # Create random input if no sample available
        logger.warning("No sample image found, using random input for validation")
        tensor = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)

    numpy_input = tensor.numpy()
    return tensor, numpy_input


def export_onnx(
    model: PneumoniaClassifier,
    cfg: DictConfig,
    output_path: Path,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained PneumoniaClassifier model.
        cfg: Hydra configuration.
        output_path: Path to save ONNX model.

    Returns:
        Path to exported ONNX model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Move model to CPU for export (ONNX export requires CPU)
    model = model.cpu()

    # Create sample input for graph tracing (values don't matter, only shape)
    image_size = cfg.data.image_size
    trace_input = torch.randn(1, 3, image_size, image_size, device="cpu")

    # Export to ONNX
    logger.info(f"Exporting ONNX model to: {output_path}")
    torch.onnx.export(
        model,
        trace_input,
        str(output_path),
        export_params=True,
        opset_version=cfg.onnx.opset_version,
        do_constant_folding=True,
        input_names=[cfg.onnx.input_name],
        output_names=[cfg.onnx.output_name],
        dynamic_axes=None,  # Fixed shape for simplicity
    )

    logger.info("ONNX export complete")
    return output_path


def validate_onnx_structure(onnx_path: Path) -> bool:
    """Validate ONNX model structure using onnx.checker.

    Args:
        onnx_path: Path to ONNX model.

    Returns:
        True if valid.

    Raises:
        onnx.checker.ValidationError: If model is invalid.
    """
    logger.info("Validating ONNX model structure...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info("ONNX model structure is valid")
    return True


def validate_onnx_numerically(
    pytorch_model: PneumoniaClassifier,
    onnx_path: Path,
    sample_tensor: torch.Tensor,
    sample_numpy: np.ndarray,
    cfg: DictConfig,
) -> dict:
    """Validate ONNX outputs match PyTorch outputs.

    Args:
        pytorch_model: Original PyTorch model.
        onnx_path: Path to ONNX model.
        sample_tensor: Sample input as PyTorch tensor.
        sample_numpy: Sample input as NumPy array.
        cfg: Configuration with tolerance settings.

    Returns:
        Validation report dict.
    """
    logger.info("Validating ONNX model numerically...")

    # PyTorch inference
    with torch.no_grad():
        pytorch_logit = pytorch_model(sample_tensor).numpy()
    pytorch_prob = 1 / (1 + np.exp(-pytorch_logit))
    pytorch_pred = int(float(pytorch_prob.item()) >= cfg.postprocess.threshold)

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_logit = session.run(None, {input_name: sample_numpy})[0]
    onnx_prob = 1 / (1 + np.exp(-onnx_logit))
    onnx_pred = int(float(onnx_prob.item()) >= cfg.postprocess.threshold)

    # Compare
    abs_diff = float(np.abs(pytorch_logit - onnx_logit).max())
    max_val = np.abs(pytorch_logit).max() + 1e-8
    rel_diff = float(np.abs(pytorch_logit - onnx_logit).max() / max_val)

    rtol = cfg.validation.rtol
    atol = cfg.validation.atol
    is_close = np.allclose(pytorch_logit, onnx_logit, rtol=rtol, atol=atol)
    labels_match = pytorch_pred == onnx_pred

    report = {
        "pytorch_logit": float(pytorch_logit.item()),
        "pytorch_prob": float(pytorch_prob.item()),
        "pytorch_pred": pytorch_pred,
        "onnx_logit": float(onnx_logit.item()),
        "onnx_prob": float(onnx_prob.item()),
        "onnx_pred": onnx_pred,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "rtol": rtol,
        "atol": atol,
        "is_close": bool(is_close),
        "labels_match": labels_match,
        "valid": bool(is_close and labels_match),
    }

    if report["valid"]:
        logger.info(f"ONNX validation PASSED (abs_diff={abs_diff:.6f})")
    else:
        logger.warning(
            f"ONNX validation FAILED (abs_diff={abs_diff:.6f}, "
            f"labels_match={labels_match})"
        )

    return report


def save_metadata(cfg: DictConfig, output_path: Path) -> Path:
    """Save preprocessing and inference metadata.

    Args:
        cfg: Hydra configuration.
        output_path: Path to save metadata JSON.

    Returns:
        Path to metadata file.
    """
    metadata = {
        "image_size": cfg.data.image_size,
        "channels": 3,
        "normalize": {
            "mean": list(cfg.data.normalize.mean),
            "std": list(cfg.data.normalize.std),
        },
        "threshold": cfg.postprocess.threshold,
        "labels": {str(k): v for k, v in cfg.postprocess.labels.items()},
        "input_name": cfg.onnx.input_name,
        "output_name": cfg.onnx.output_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to: {output_path}")
    return output_path


def check_tensorrt_available() -> bool:
    """Check if TensorRT (trtexec) is available."""
    return shutil.which("trtexec") is not None


def convert_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
) -> Path | None:
    """Convert ONNX model to TensorRT engine using trtexec.

    Args:
        onnx_path: Path to ONNX model.
        engine_path: Path to save TensorRT engine.
        fp16: Use FP16 precision if supported.

    Returns:
        Path to engine file if successful, None otherwise.
    """
    if not check_tensorrt_available():
        logger.error(
            "TensorRT (trtexec) is not available. "
            "Install TensorRT or skip TensorRT conversion."
        )
        return None

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]

    if fp16:
        cmd.append("--fp16")

    logger.info(f"Converting to TensorRT: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("TensorRT conversion complete")
        logger.debug(result.stdout)
        return engine_path
    except subprocess.CalledProcessError as e:
        logger.error(f"TensorRT conversion failed: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("trtexec not found in PATH")
        return None


def validate_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    sample_numpy: np.ndarray,
    cfg: DictConfig,
) -> dict | None:
    """Validate TensorRT outputs match ONNX outputs.

    Args:
        onnx_path: Path to ONNX model.
        engine_path: Path to TensorRT engine.
        sample_numpy: Sample input as NumPy array.
        cfg: Configuration with tolerance settings.

    Returns:
        Validation report dict, or None if TensorRT not available.
    """
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError:
        logger.warning(
            "TensorRT Python bindings not available. Skipping TensorRT validation."
        )
        return None

    logger.info("Validating TensorRT engine...")

    # ONNX inference for reference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_logit = session.run(None, {input_name: sample_numpy})[0]
    onnx_prob = 1 / (1 + np.exp(-onnx_logit))
    onnx_pred = int(onnx_prob >= cfg.postprocess.threshold)

    # TensorRT inference
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    output_shape = (1,)

    d_input = cuda.mem_alloc(sample_numpy.nbytes)
    d_output = cuda.mem_alloc(np.empty(output_shape, dtype=np.float32).nbytes)

    stream = cuda.Stream()

    # Copy input
    cuda.memcpy_htod_async(d_input, sample_numpy.astype(np.float32), stream)

    # Run inference
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle,
    )

    # Copy output
    trt_logit = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(trt_logit, d_output, stream)
    stream.synchronize()

    trt_prob = 1 / (1 + np.exp(-trt_logit))
    trt_pred = int(trt_prob >= cfg.postprocess.threshold)

    # Compare
    abs_diff = float(np.abs(onnx_logit.flatten() - trt_logit).max())
    rel_diff = float(
        np.abs(onnx_logit.flatten() - trt_logit).max()
        / (np.abs(onnx_logit).max() + 1e-8)
    )

    rtol = cfg.validation.tensorrt_rtol
    atol = cfg.validation.tensorrt_atol
    is_close = np.allclose(onnx_logit.flatten(), trt_logit, rtol=rtol, atol=atol)
    labels_match = onnx_pred == trt_pred

    report = {
        "onnx_logit": float(onnx_logit.item()),
        "onnx_prob": float(onnx_prob.item()),
        "onnx_pred": onnx_pred,
        "tensorrt_logit": float(trt_logit.item()),
        "tensorrt_prob": float(trt_prob.item()),
        "tensorrt_pred": trt_pred,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "rtol": rtol,
        "atol": atol,
        "is_close": bool(is_close),
        "labels_match": labels_match,
        "valid": bool(is_close and labels_match),
    }

    if report["valid"]:
        logger.info(f"TensorRT validation PASSED (abs_diff={abs_diff:.6f})")
    else:
        logger.warning(
            f"TensorRT validation FAILED (abs_diff={abs_diff:.6f}, "
            f"labels_match={labels_match})"
        )

    return report


def run_export(cfg: DictConfig) -> dict:
    """Run full export pipeline.

    Args:
        cfg: Hydra configuration.

    Returns:
        Export results dict.
    """
    results = {
        "onnx_path": None,
        "tensorrt_path": None,
        "metadata_path": None,
        "onnx_validation": None,
        "tensorrt_validation": None,
    }

    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / cfg.checkpoint_path
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run training first: python -m pneumonia_xray.commands train")
        return results

    logger.info(f"Loading model from: {checkpoint_path}")
    # Load model to CPU for ONNX export (avoids device mismatch issues)
    model = PneumoniaClassifier.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu"
    )
    model.train(False)  # Set to eval mode

    # Prepare output directory
    output_dir = PROJECT_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample image for validation
    samples_dir = PROJECT_ROOT / cfg.samples_dir
    sample_path = None
    if samples_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            samples = list(samples_dir.glob(ext))
            if samples:
                sample_path = samples[0]
                break

    # Load sample input
    sample_tensor, sample_numpy = load_sample_input(cfg, sample_path)

    # Export ONNX
    onnx_path = PROJECT_ROOT / cfg.onnx.path
    export_onnx(model, cfg, onnx_path)
    results["onnx_path"] = str(onnx_path)

    # Validate ONNX structure
    validate_onnx_structure(onnx_path)

    # Validate ONNX numerically
    onnx_report = validate_onnx_numerically(
        model, onnx_path, sample_tensor, sample_numpy, cfg
    )
    results["onnx_validation"] = onnx_report

    # Save ONNX validation report
    onnx_val_path = output_dir / "validation_onnx.json"
    with open(onnx_val_path, "w") as f:
        json.dump(onnx_report, f, indent=2)
    logger.info(f"Saved ONNX validation report: {onnx_val_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    save_metadata(cfg, metadata_path)
    results["metadata_path"] = str(metadata_path)

    # Convert to TensorRT
    if check_tensorrt_available():
        engine_path = PROJECT_ROOT / cfg.tensorrt.path
        trt_result = convert_to_tensorrt(onnx_path, engine_path, fp16=cfg.tensorrt.fp16)
        if trt_result:
            results["tensorrt_path"] = str(engine_path)

            # Validate TensorRT
            trt_report = validate_tensorrt(onnx_path, engine_path, sample_numpy, cfg)
            if trt_report:
                results["tensorrt_validation"] = trt_report
                trt_val_path = output_dir / "validation_tensorrt.json"
                with open(trt_val_path, "w") as f:
                    json.dump(trt_report, f, indent=2)
                logger.info(f"Saved TensorRT validation report: {trt_val_path}")
    else:
        logger.warning(
            "TensorRT (trtexec) not available. Skipping TensorRT conversion. "
            "Install TensorRT to enable engine generation."
        )

    return results
