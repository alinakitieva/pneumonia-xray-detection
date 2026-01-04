# Pneumonia X-ray Detection

Binary classification of chest X-ray images to detect pneumonia using deep learning.

## Problem Description

**Task:** Classify chest X-ray images as either **Pneumonia** or **Normal**.

**Input:** Chest X-ray image (grayscale, resized to 224x224)

**Output:** Probability score (0-1) and predicted label (Pneumonia/Normal)

**Metrics:** Accuracy, Precision, Recall, F1-Score

## Dataset

**Chest X-Ray Images (Pneumonia)** - 5,856 images organized into train/val/test splits.

| Split | Normal | Pneumonia | Total |
| ----- | ------ | --------- | ----- |
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

Data is tracked with DVC and stored in Cloudflare R2.

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

### 1. Install dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-username/pneumonia-xray-detection.git
cd pneumonia-xray-detection

# Install all dependencies
uv sync --all-extras
```

### 2. Download dataset

```bash
# Pull data from DVC remote
uv run dvc pull
```

This downloads ~1.2 GB of chest X-ray images to `data/raw/chest_xray/`.

### 3. Verify installation

```bash
# Check package
uv run python -c "import pneumonia_xray; print(pneumonia_xray.__version__)"

# Check data
ls data/raw/chest_xray/
```

## Usage

### Training

```bash
# Train with default config
uv run python -m pneumonia_xray.commands train

# Train with config overrides
uv run python -m pneumonia_xray.commands train trainer.max_epochs=5
uv run python -m pneumonia_xray.commands train data.batch_size=64

# View resolved config
uv run python -m pneumonia_xray.commands train --cfg job
```

Training will:

- Load data from `data/raw/chest_xray/`
- Train ResNet-18 with pretrained ImageNet weights
- Log metrics to MLflow (if server running)
- Save best checkpoint to `artifacts/checkpoints/best.ckpt`
- Generate plots in `plots/<run_id>/` (confusion matrix, ROC curve)

### Inference

```bash
# Single image
uv run python -m pneumonia_xray.commands infer input.path=path/to/image.png

# Folder of images
uv run python -m pneumonia_xray.commands infer input.path=path/to/folder/

# Example with test data
uv run python -m pneumonia_xray.commands infer input.path=data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg
```

Output format:

```
path=data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg prob=0.2104 pred=NORMAL
```

### Model Export (ONNX/TensorRT)

```bash
# Export trained model to ONNX (and TensorRT if available)
uv run python -m pneumonia_xray.commands export

# Export with FP32 TensorRT (instead of FP16)
uv run python -m pneumonia_xray.commands export tensorrt.fp16=false
```

Export generates:

- `artifacts/export/model.onnx` - ONNX model for portable inference
- `artifacts/export/metadata.json` - Preprocessing config (normalization, threshold)
- `artifacts/export/validation_onnx.json` - Numerical validation report
- `artifacts/export/model.trt` - TensorRT engine (if trtexec available)

### MLflow Experiment Tracking

```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 8080

# Run training (logs automatically)
uv run python -m pneumonia_xray.commands train
```

MLflow logs:

- Metrics: train_loss, val_loss, accuracy, precision, recall, F1
- Hyperparameters: all config values
- Git commit ID
- Artifacts: plots, config.yaml

## Model Architecture

- **Backbone:** ResNet-18 (pretrained on ImageNet)
- **Head:** Linear layer (512 -> 1)
- **Loss:** BCEWithLogitsLoss with pos_weight=2.9 (handles class imbalance)
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau

## Configuration

All hyperparameters are managed via Hydra YAML configs:

```
configs/
├── train.yaml           # Main training config
├── infer.yaml           # Main inference config
├── export.yaml          # Model export config
├── data/default.yaml    # Data loading & augmentation
├── model/resnet18.yaml  # Model architecture
├── trainer/default.yaml # Training settings
├── logging/mlflow.yaml  # MLflow settings
└── postprocess/default.yaml  # Inference thresholds
```

Override any parameter via CLI:

```bash
uv run python -m pneumonia_xray.commands train trainer.learning_rate=0.001 trainer.max_epochs=20
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run pre-commit hooks
uv run pre-commit run -a

# Run tests
uv run pytest
```

## Tech Stack

| Component             | Tool                |
| --------------------- | ------------------- |
| Training              | PyTorch Lightning   |
| Model                 | ResNet-18           |
| Metrics               | torchmetrics        |
| Config                | Hydra               |
| Experiment tracking   | MLflow              |
| Model export          | ONNX, TensorRT      |
| Data versioning       | DVC + Cloudflare R2 |
| Dependency management | uv                  |
| Code quality          | ruff, pre-commit    |
