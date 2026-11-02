"""Shared fixtures for API regression tests."""

import pandas as pd
import pytest

from src.schemas import ChurnPredictionRequest


@pytest.fixture
def request_payload() -> dict:
    """Return a representative raw Telco payload using API field aliases."""
    return {
        "City": "Los Angeles",
        "Zip Code": 90001,
        "Latitude": 33.9736,
        "Longitude": -118.2490,
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 75.5,
        "Total Charges": 906.0,
        "CLTV": 5000,
    }


@pytest.fixture
def encoded_features() -> pd.DataFrame:
    """Return a zero-valued frame with the champion's expected columns."""
    from src.utils.parameter_encoding import ENCODED_COLUMNS

    return pd.DataFrame([[0] * len(ENCODED_COLUMNS)], columns=ENCODED_COLUMNS)


@pytest.fixture
def request_model(request_payload: dict) -> ChurnPredictionRequest:
    """Return the validated request model for encoder unit tests."""
    return ChurnPredictionRequest.model_validate(request_payload)
