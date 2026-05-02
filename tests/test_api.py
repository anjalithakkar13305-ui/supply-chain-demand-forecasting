"""
tests/test_api.py
-----------------
Integration tests for the FastAPI endpoints.
Uses FastAPI's TestClient — no running server needed.
Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Skip all API tests if models haven't been trained yet
pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "models" / "best_model.pkl").exists(),
    reason="Models not trained yet. Run python src/train_model.py first.",
)


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_model_info(self, client):
        response = client.get("/health")
        data = response.json()
        assert "model_ready" in data
        assert "best_model" in data
        assert "timestamp" in data

    def test_health_model_is_ready(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_ready"] is True


# ── Forecast product endpoint ─────────────────────────────────────────────────

class TestForecastProductEndpoint:
    def test_valid_request_returns_200(self, client):
        payload = {"product_name": "Field & Stream Sportsman 16 Gun Fire Safe",
                   "horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        # Either 200 (found) or 404 (product not in dataset) is acceptable
        assert response.status_code in [200, 404]

    def test_response_has_forecast_list(self, client):
        payload = {"product_name": "Field & Stream Sportsman 16 Gun Fire Safe",
                   "horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "forecast" in data
            assert isinstance(data["forecast"], list)

    def test_forecast_points_have_required_fields(self, client):
        payload = {"product_name": "Field & Stream Sportsman 16 Gun Fire Safe",
                   "horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data["forecast"]:
                point = data["forecast"][0]
                assert "date" in point
                assert "predicted_demand" in point
                assert "lower_bound" in point
                assert "upper_bound" in point

    def test_predicted_demand_is_non_negative(self, client):
        payload = {"product_name": "Field & Stream Sportsman 16 Gun Fire Safe",
                   "horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        if response.status_code == 200:
            data = response.json()
            for point in data["forecast"]:
                assert point["predicted_demand"] >= 0

    def test_lower_bound_lte_predicted(self, client):
        payload = {"product_name": "Field & Stream Sportsman 16 Gun Fire Safe",
                   "horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        if response.status_code == 200:
            data = response.json()
            for point in data["forecast"]:
                assert point["lower_bound"] <= point["predicted_demand"]

    def test_invalid_horizon_returns_422(self, client):
        # horizon_days must be between 7 and 180
        payload = {"product_name": "Some Product", "horizon_days": 500}
        response = client.post("/forecast/product", json=payload)
        assert response.status_code == 422

    def test_missing_product_name_returns_422(self, client):
        payload = {"horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        assert response.status_code == 422

    def test_unknown_product_returns_404(self, client):
        payload = {"product_name": "NONEXISTENT_PRODUCT_XYZ_123",
                   "horizon_days": 30}
        response = client.post("/forecast/product", json=payload)
        assert response.status_code == 404


# ── Bulk forecast endpoint ────────────────────────────────────────────────────

class TestBulkForecastEndpoint:
    def test_valid_request_returns_200(self, client):
        payload = {"category_name": "Sporting Goods", "horizon_days": 30}
        response = client.post("/forecast/bulk", json=payload)
        assert response.status_code in [200, 404]

    def test_response_has_products_list(self, client):
        payload = {"category_name": "Sporting Goods", "horizon_days": 30}
        response = client.post("/forecast/bulk", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "products" in data
            assert isinstance(data["products"], list)

    def test_products_sorted_by_demand_desc(self, client):
        payload = {"category_name": "Sporting Goods", "horizon_days": 30}
        response = client.post("/forecast/bulk", json=payload)
        if response.status_code == 200:
            data = response.json()
            demands = [p["total_predicted_demand"] for p in data["products"]]
            assert demands == sorted(demands, reverse=True)

    def test_unknown_category_returns_404(self, client):
        payload = {"category_name": "FAKE_CATEGORY_99999", "horizon_days": 30}
        response = client.post("/forecast/bulk", json=payload)
        assert response.status_code == 404

    def test_horizon_too_large_returns_422(self, client):
        payload = {"category_name": "Sporting Goods", "horizon_days": 200}
        response = client.post("/forecast/bulk", json=payload)
        assert response.status_code == 422
