import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from pneumonia_xray.export import run_export
from pneumonia_xray.infer import run_inference

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the pneumonia detection model."""
    from pneumonia_xray.train import train_model

    checkpoint_path = train_model(cfg)
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


@hydra.main(config_path=CONFIG_PATH, config_name="infer", version_base=None)
def infer(cfg: DictConfig) -> None:
    """Run inference on X-ray images."""

    results = run_inference(cfg)
    if results:
        print(f"\nProcessed {len(results)} image(s)")


@hydra.main(config_path=CONFIG_PATH, config_name="export", version_base=None)
def export(cfg: DictConfig) -> None:
    """Export model to ONNX and TensorRT formats."""

    results = run_export(cfg)

    print("\n=== Export Results ===")
    if results["onnx_path"]:
        print(f"ONNX model: {results['onnx_path']}")
    if results["metadata_path"]:
        print(f"Metadata: {results['metadata_path']}")
    if results["tensorrt_path"]:
        print(f"TensorRT engine: {results['tensorrt_path']}")
    else:
        print("TensorRT: skipped (trtexec not available)")

    if results["onnx_validation"]:
        ov = results["onnx_validation"]
        status = "PASSED" if ov["valid"] else "FAILED"
        print(f"ONNX validation: {status} (diff={ov['abs_diff']:.6f})")

    if results["tensorrt_validation"]:
        tv = results["tensorrt_validation"]
        status = "PASSED" if tv["valid"] else "FAILED"
        print(f"TensorRT validation: {status} (diff={tv['abs_diff']:.6f})")


@hydra.main(config_path=CONFIG_PATH, config_name="serve", version_base=None)
def serve(cfg: DictConfig) -> None:
    """Prepare model repository for Triton serving."""
    from pneumonia_xray.serve import run_prepare_serving

    results = run_prepare_serving(cfg)

    print("\n=== Triton Model Repository ===")
    if results["model_repository"]:
        print(f"Repository: {results['model_repository']}")
        print(f"Model: {results['model_path']}")
        print(f"Config: {results['config_path']}")
        print()
        print("To start Triton server:")
        print(
            f"  docker run --rm -p 8000:8000 -p 8001:8001 "
            f"-v {results['model_repository']}:/models "
            f"nvcr.io/nvidia/tritonserver:24.01-py3 "
            f"tritonserver --model-repository=/models"
        )
    else:
        print("Failed to prepare model repository")


@hydra.main(config_path=CONFIG_PATH, config_name="serve", version_base=None)
def triton_infer(cfg: DictConfig) -> None:
    """Run inference via Triton Inference Server."""
    from pneumonia_xray.serve import run_triton_inference

    results = run_triton_inference(cfg)
    if results:
        print(f"\nProcessed {len(results)} image(s) via Triton")


def main() -> None:
    """Main CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Pneumonia X-ray Detection CLI")
        print("Available commands: train, infer, export, serve, triton_infer")
        print("Usage: python -m pneumonia_xray.commands <command> [overrides...]")
        print()
        print("Examples:")
        print("  python -m pneumonia_xray.commands train")
        print("  python -m pneumonia_xray.commands train trainer.max_epochs=5")
        print("  python -m pneumonia_xray.commands infer input.path=/path/to/image.png")
        print("  python -m pneumonia_xray.commands export")
        print("  python -m pneumonia_xray.commands serve")
        print("  python -m pneumonia_xray.commands triton_infer input.path=image.png")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        train()
    elif command == "infer":
        infer()
    elif command == "export":
        export()
    elif command == "serve":
        serve()
    elif command == "triton_infer":
        triton_infer()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
