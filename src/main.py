"""
Telco Churn API
Local deployment
"""
from fastapi import FastAPI

from src.core import API_VERSION
from src.routers.info import router as info_router
from src.routers.predict import router as predict_router

app = FastAPI(
    title="Telco Churn API",
    description="""
## API for predicting customer churn for Telco - The Telephone Company

### Main endpoints
- `POST /predict` - Individual churn prediction
- `GET /health` - Health check
""",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(info_router)
app.include_router(predict_router)