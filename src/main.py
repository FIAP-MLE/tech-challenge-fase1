"""
Telco Churn API
Local deployment
"""
import logging

from fastapi import FastAPI
from src.core import API_VERSION
from src.routers.info import router as info_router
from src.routers.predict import router as predict_router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fast API
app = FastAPI(
    title = "Telco Churn API",
    description = """
## API for predicting customer churn for Telco - The Telephone Company

### Main endpoints
- `POST /predict` - Predicao individual
- `GET /health` - Health check
""",
    version = API_VERSION,
    docs_url = "/docs",
    redoc_url = "/redoc"
)

app.include_router(info_router)
app.include_router(predict_router)