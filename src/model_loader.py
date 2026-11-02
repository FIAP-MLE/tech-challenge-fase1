"""
Model Loader

Loads the model and tells the api the model is healthy and ready
"""
from pathlib import Path
from src.core import logger
import joblib

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# Resolve the artifact from the repository root instead of from the process
# working directory, which can vary between PyCharm, Uvicorn, and CI.
MODEL_PATHS = [
    PROJECT_DIR / "models" / "telco_random_forest_champion.joblib",
]

model = None

for model_path in MODEL_PATHS:
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = joblib.load(f)
            logger.info("model_loaded", extra={"path": str(model_path)})
            break

MODEL_LOADED = model is not None

if not MODEL_LOADED:
    logger.warning(
        "model_not_found",
        extra={"paths": [str(path) for path in MODEL_PATHS]}
    )