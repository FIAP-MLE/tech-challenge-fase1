"""
Prediction route for the Telco churn API.
"""

from fastapi import APIRouter, HTTPException, status

from src.core import logger
from src.model_loader import MODEL_LOADED, model
from src.schemas import ChurnPredictionRequest, ChurnPredictionResponse
from src.utils.parameter_encoding import encode_parameters


router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(request: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Predict whether a Telco customer will churn."""
    if not MODEL_LOADED or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available",
        )

    try:
        encoded_features = encode_parameters(request)
        prediction = int(model.predict(encoded_features)[0])
        churn_probability = float(model.predict_proba(encoded_features)[0][1])
    except ValueError as exc:
        logger.warning("prediction_validation_failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("prediction_failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed",
        ) from exc

    logger.info(
        "prediction_completed",
        extra={
            "prediction": prediction,
            "churn_probability": churn_probability,
        },
    )
    return ChurnPredictionResponse(
        churn_prediction=prediction,
        churn_label="Churn" if prediction else "No churn",
        churn_probability=churn_probability,
        model="telco_random_forest_champion",
    )
