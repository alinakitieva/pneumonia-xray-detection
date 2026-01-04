# Pneumonia X-ray Detection

Binary classification of chest X-ray images to detect pneumonia using deep learning.

## Problem Description

**Task:** Classify chest X-ray images as either **Pneumonia** or **Normal**.

**Input:** Chest X-ray image (grayscale, resized to 224x224)

**Output:** Probability score and predicted label (Pneumonia/Normal)

**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

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
uv run python -m pneumonia_xray.commands train
```

This will:

- Load data from `data/raw/chest_xray/`
- Train ResNet-18 with pretrained ImageNet weights
- Save best checkpoint to `artifacts/checkpoints/best.ckpt`
- Generate plots in `plots/` (confusion matrix, ROC curve)

### Inference

```bash
uv run python -m pneumonia_xray.commands infer
```

## Model Architecture

- **Backbone:** ResNet-18 (pretrained on ImageNet)
- **Head:** Linear layer (512 -> 1)
- **Loss:** BCEWithLogitsLoss with pos_weight=2.9 (handles class imbalance)
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau

## Production

### Export to ONNX

```bash
# Export trained model to ONNX format
uv run python -m pneumonia_xray.commands export --format onnx
```

### Convert to TensorRT

```bash
# Convert ONNX to TensorRT engine
uv run python -m pneumonia_xray.commands export --format tensorrt
```

### Serving

```bash
# Start MLflow model server
mlflow models serve -m runs:/<run_id>/model -p 5000
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
| Data versioning       | DVC + Cloudflare R2 |
| Export                | ONNX, TensorRT      |
| Dependency management | uv                  |
| Code quality          | ruff, pre-commit    |
