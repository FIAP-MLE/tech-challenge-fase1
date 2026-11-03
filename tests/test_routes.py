"""Regression tests for the public FastAPI endpoints."""

import asyncio

import httpx

from src.main import app
from src.routers import predict as predict_router


def request(method: str, path: str, **kwargs) -> httpx.Response:
    """Execute an ASGI request without the deprecated TestClient adapter."""

    async def send_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


def test_home_describes_prediction_contract():
    response = request("GET", "/")

    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "Telco Churn Predictor"
    assert len(body["expected_features"]) == 24
    assert "Churn Score" not in body["expected_features"]


def test_health_reports_model_state():
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "degraded"}


def test_predict_returns_churn_result(
    request_payload, encoded_features, monkeypatch
):
    class FakeModel:
        def predict(self, features):
            assert list(features.columns) == list(encoded_features.columns)
            return [1]

        def predict_proba(self, features):
            return [[0.2, 0.8]]

    monkeypatch.setattr(predict_router, "MODEL_LOADED", True)
    monkeypatch.setattr(predict_router, "model", FakeModel())
    monkeypatch.setattr(
        predict_router,
        "encode_parameters",
        lambda request: encoded_features,
    )

    response = request("POST", "/predict", json=request_payload)

    assert response.status_code == 200
    assert response.json() == {
        "churn_prediction": 1,
        "churn_label": "Churn",
        "churn_probability": 0.8,
        "model": "telco_random_forest_champion",
    }


def test_predict_returns_validation_error_for_unknown_city(
    request_payload, monkeypatch
):
    monkeypatch.setattr(
        predict_router,
        "encode_parameters",
        lambda request: (_ for _ in ()).throw(
            ValueError("City 'Unknown' was not present in the training dataset")
        ),
    )

    response = request("POST", "/predict", json=request_payload)

    assert response.status_code == 422
    assert "not present in the training dataset" in response.json()["detail"]


def test_predict_returns_service_unavailable_when_model_is_missing(
    request_payload, monkeypatch
):
    monkeypatch.setattr(predict_router, "MODEL_LOADED", False)
    monkeypatch.setattr(predict_router, "model", None)

    response = request("POST", "/predict", json=request_payload)

    assert response.status_code == 503
