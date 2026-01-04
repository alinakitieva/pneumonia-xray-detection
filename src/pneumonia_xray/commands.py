import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

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


def main() -> None:
    """Main CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Pneumonia X-ray Detection CLI")
        print("Available commands: train, infer")
        print("Usage: python -m pneumonia_xray.commands <command> [overrides...]")
        print()
        print("Examples:")
        print("  python -m pneumonia_xray.commands train")
        print("  python -m pneumonia_xray.commands train trainer.max_epochs=5")
        print("  python -m pneumonia_xray.commands infer input.path=/path/to/image.png")
        print("  python -m pneumonia_xray.commands infer input.path=/path/to/folder")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        train()
    elif command == "infer":
        infer()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
