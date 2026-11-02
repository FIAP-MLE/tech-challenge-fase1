"""
Information and health routers
"""
from fastapi import APIRouter
from src.core import API_VERSION, ENVIRONMENT
from src.model_loader import MODEL_LOADED

router = APIRouter(tags=["Info"])

@router.get("/")
def home():
    """API Information"""
    return {
        "api": "Telco Chrun Predictor",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "model_loaded": MODEL_LOADED,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
        },
        "expected_features": [
            "City",
            "Zip Code",
            "Latitude",
            "Longitude",
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Tenure Months",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Contract",
            "Paperless Billing",
            "Payment Method",
            "Monthly Charges",
            "Total Charges",
            "CLTV",
        ],
    }

@router.get("/health")
def health():
    """Health check for monitoring purposes"""
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "environment": ENVIRONMENT,
        "version": API_VERSION,
    }
