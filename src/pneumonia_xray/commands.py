import sys

from pneumonia_xray.train import train_model


def train() -> None:
    """Train the pneumonia detection model."""

    checkpoint_path = train_model()
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


def infer() -> None:
    """Run inference on X-ray images."""
    pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pneumonia X-ray Detection CLI")
        print("Available commands: train, infer")
        print("Usage: python -m pneumonia_xray.commands <command>")
        sys.exit(1)

    command = sys.argv[1]
    if command == "train":
        train()
    elif command == "infer":
        infer()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
