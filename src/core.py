"""
Global API configuration
"""

import os
from src.logging_config import setup_logging

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = "1.0.0"

logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))