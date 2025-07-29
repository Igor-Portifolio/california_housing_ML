import pandas as pd
import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_housing_data(filepath: str = None) -> pd.DataFrame:

    if filepath is None:
        filepath = DATA_DIR / "housing.csv"

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at {filepath}")

        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        print(f"Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def save_data(df: pd.DataFrame, filename: str, subfolder: str = "processed"):

    save_dir = DATA_DIR / subfolder
    os.makedirs(save_dir, exist_ok=True)

    filepath = save_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def save_model(model, filename: str, subfolder: str = "trained"):

    save_dir = MODELS_DIR / subfolder
    os.makedirs(save_dir, exist_ok=True)

    filepath = save_dir / filename
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename: str, subfolder: str = "trained"):

    filepath = MODELS_DIR / subfolder / filename

    if not filepath.exists():
        raise FileNotFoundError(f"No model file found at {filepath}")

    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def list_saved_models(subfolder: str = "trained") -> list:
    models_dir = MODELS_DIR / subfolder
    if not models_dir.exists():
        return []

    return [f.name for f in models_dir.glob("*.pkl")]
