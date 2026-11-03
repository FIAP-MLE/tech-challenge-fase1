"""
Request and response schemas for the Telco churn prediction API.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


NonNegativeFloat = Annotated[float, Field(ge=0)]
BinaryCategory = Literal["Yes", "No"]


class ChurnPredictionRequest(BaseModel):
    """Raw Telco customer features accepted by the prediction endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    city: str = Field(
        alias="City",
        min_length=1,
        description="Customer city. It must exist in the training dataset.",
    )
    zip_code: int = Field(
        alias="Zip Code",
        ge=0,
        description="Customer postal code.",
    )
    latitude: float = Field(
        alias="Latitude",
        ge=-90,
        le=90,
        description="Customer latitude in decimal degrees.",
    )
    longitude: float = Field(
        alias="Longitude",
        ge=-180,
        le=180,
        description="Customer longitude in decimal degrees.",
    )
    gender: Literal["Male", "Female"] = Field(
        alias="Gender",
        description="Customer gender.",
    )
    senior_citizen: BinaryCategory = Field(
        alias="Senior Citizen",
        description="Whether the customer is 65 or older.",
    )
    partner: BinaryCategory = Field(
        alias="Partner",
        description="Whether the customer has a partner.",
    )
    dependents: BinaryCategory = Field(
        alias="Dependents",
        description="Whether the customer has dependents.",
    )
    tenure_months: NonNegativeFloat = Field(
        alias="Tenure Months",
        description="Number of months the customer has been subscribed.",
    )
    phone_service: BinaryCategory = Field(
        alias="Phone Service",
        description="Whether the customer has phone service.",
    )
    multiple_lines: Literal["No", "Yes", "No phone service"] = Field(
        alias="Multiple Lines",
        description="Whether the customer has multiple phone lines.",
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        alias="Internet Service",
        description="Internet service type.",
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        alias="Online Security",
        description="Whether the customer has online security.",
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        alias="Online Backup",
        description="Whether the customer has online backup.",
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        alias="Device Protection",
        description="Whether the customer has device protection.",
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        alias="Tech Support",
        description="Whether the customer has technical support.",
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        alias="Streaming TV",
        description="Whether the customer has streaming TV.",
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        alias="Streaming Movies",
        description="Whether the customer has streaming movies.",
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        alias="Contract",
        description="Customer contract term.",
    )
    paperless_billing: BinaryCategory = Field(
        alias="Paperless Billing",
        description="Whether the customer uses paperless billing.",
    )
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(
        alias="Payment Method",
        description="Payment method used by the customer.",
    )
    monthly_charges: NonNegativeFloat = Field(
        alias="Monthly Charges",
        description="Current monthly charge in the dataset currency.",
    )
    total_charges: NonNegativeFloat = Field(
        alias="Total Charges",
        description="Total amount charged to the customer to date.",
    )
    cltv: NonNegativeFloat = Field(
        alias="CLTV",
        description="Customer lifetime value score provided by the source dataset.",
    )


class ChurnPredictionResponse(BaseModel):
    """Prediction returned by the champion model."""

    churn_prediction: int = Field(
        ge=0,
        le=1,
        description="Binary prediction: 1 for churn and 0 for no churn.",
    )
    churn_label: str = Field(
        description="Human-readable version of the binary prediction.",
    )
    churn_probability: float = Field(
        ge=0,
        le=1,
        description="Predicted probability that the customer will churn.",
    )
    model: str = Field(
        description="Identifier of the model artifact that produced the result.",
    )


# Backward-compatible alias for the original misspelling.
ChrunPredictionResponse = ChurnPredictionResponse
