# pneumonia-xray-detection

A pneumonia screening assistant that classifies chest X-ray images as Pneumonia or Normal using deep learning.


## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-username/pneumonia-xray-detection.git
cd pneumonia-xray-detection

# Install all dependencies (runtime + dev)
uv sync --all-extras

# Or install only runtime dependencies
uv sync
```

## Verify Installation

```bash
# Check package imports correctly
uv run python -c "import pneumonia_xray; print(pneumonia_xray.__version__)"

# Run the CLI entrypoint
uv run python -m pneumonia_xray.commands
```

## Usage

```bash
# Train the model
uv run python -m pneumonia_xray.commands train

# Run inference
uv run python -m pneumonia_xray.commands infer
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
