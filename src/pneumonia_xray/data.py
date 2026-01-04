import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHEST_XRAY_DIR = RAW_DATA_DIR / "chest_xray"


def download_data() -> Path:
    """Download dataset via DVC if not present.

    Returns:
        Path to the chest_xray dataset directory.

    Raises:
        RuntimeError: If DVC pull fails.
    """
    if not CHEST_XRAY_DIR.exists() or not any(CHEST_XRAY_DIR.iterdir()):
        print("Data not found. Pulling from DVC remote...")
        result = subprocess.run(
            ["dvc", "pull"],
            cwd=DATA_DIR.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"DVC pull failed: {result.stderr}")
        print("Data downloaded successfully.")
    else:
        print(f"Data already present at {CHEST_XRAY_DIR}")

    return CHEST_XRAY_DIR


def get_data_path() -> Path:
    """Get path to dataset, downloading if necessary."""
    return download_data()
