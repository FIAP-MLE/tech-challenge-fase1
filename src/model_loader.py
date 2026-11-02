"""
Load the persisted champion model used by the API.
"""

from pathlib import Path

import joblib

from src.core import logger


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "telco_random_forest_champion.joblib"


def load_model() -> object | None:
    """Load the champion artifact using a path independent of the CWD."""
    if not MODEL_PATH.exists():
        logger.warning("model_not_found", extra={"path": str(MODEL_PATH)})
        return None

    try:
        model = joblib.load(MODEL_PATH)
    except (OSError, ValueError, ImportError) as exc:
        logger.exception(
            "model_load_failed",
            extra={"path": str(MODEL_PATH), "error": str(exc)},
        )
        return None

    logger.info("model_loaded", extra={"path": str(MODEL_PATH)})
    return model


model = load_model()
MODEL_LOADED = model is not None