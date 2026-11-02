"""Tests for request and response validation contracts."""

import pytest
from pydantic import ValidationError

from src.schemas import ChurnPredictionRequest, ChurnPredictionResponse


def test_request_accepts_raw_dataset_aliases(request_payload):
    request = ChurnPredictionRequest.model_validate(request_payload)

    assert request.city == "Los Angeles"
    assert request.internet_service == "Fiber optic"


def test_request_rejects_invalid_category(request_payload):
    request_payload["Internet Service"] = "Satellite"

    with pytest.raises(ValidationError):
        ChurnPredictionRequest.model_validate(request_payload)


def test_response_rejects_probability_outside_range():
    with pytest.raises(ValidationError):
        ChurnPredictionResponse(
            churn_prediction=1,
            churn_label="Churn",
            churn_probability=1.1,
            model="test-model",
        )
