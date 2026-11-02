"""
Translate raw Telco features into the champion model's encoded feature vector.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.schemas import ChurnPredictionRequest


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_DIR / "data" / "Telco_customer_churn.xlsx"
ENCODED_COLUMNS = [
    "Zip Code",
    "Latitude",
    "Longitude",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Gender_Male",
    "Senior Citizen_Yes",
    "Partner_Yes",
    "Dependents_Yes",
    "Phone Service_Yes",
    "Multiple Lines_No phone service",
    "Multiple Lines_Yes",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No internet service",
    "Online Security_Yes",
    "Online Backup_No internet service",
    "Online Backup_Yes",
    "Device Protection_No internet service",
    "Device Protection_Yes",
    "Tech Support_No internet service",
    "Tech Support_Yes",
    "Streaming TV_No internet service",
    "Streaming TV_Yes",
    "Streaming Movies_No internet service",
    "Streaming Movies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "Paperless Billing_Yes",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Payment Method_Mailed check",
    "City_Frequency",
]

ONE_HOT_COLUMNS = {
    "Gender": ["Male"],
    "Senior Citizen": ["Yes"],
    "Partner": ["Yes"],
    "Dependents": ["Yes"],
    "Phone Service": ["Yes"],
    "Multiple Lines": ["No phone service", "Yes"],
    "Internet Service": ["Fiber optic", "No"],
    "Online Security": ["No internet service", "Yes"],
    "Online Backup": ["No internet service", "Yes"],
    "Device Protection": ["No internet service", "Yes"],
    "Tech Support": ["No internet service", "Yes"],
    "Streaming TV": ["No internet service", "Yes"],
    "Streaming Movies": ["No internet service", "Yes"],
    "Contract": ["One year", "Two year"],
    "Paperless Billing": ["Yes"],
    "Payment Method": [
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


@lru_cache(maxsize=1)
def _city_frequencies() -> pd.Series:
    """Build and cache the unsupervised city-frequency mapping used in training."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {RAW_DATA_PATH}")
    return pd.read_excel(RAW_DATA_PATH)["City"].value_counts(normalize=True)


def encode_parameters(request: ChurnPredictionRequest) -> pd.DataFrame:
    """
    Convert one raw request into the exact feature columns expected by the model.

    The returned dataframe has one row, preserves the training column order, and
    does not expose implementation details to API consumers.
    """
    raw = request.model_dump(by_alias=True)
    encoded = {
        column: raw[column]
        for column in [
            "Zip Code",
            "Latitude",
            "Longitude",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "CLTV",
        ]
    }

    for source_column, categories in ONE_HOT_COLUMNS.items():
        value = raw[source_column]
        for category in categories:
            encoded[f"{source_column}_{category}"] = int(value == category)

    city_frequency = _city_frequencies()
    if raw["City"] not in city_frequency.index:
        raise ValueError(
            f"City '{raw['City']}' was not present in the training dataset"
        )
    encoded["City_Frequency"] = float(city_frequency[raw["City"]])

    return pd.DataFrame([encoded], columns=ENCODED_COLUMNS)
