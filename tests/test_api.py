import pytest
from datetime import date
from fastapi.testclient import TestClient 
from dfw_forecasting.api.main import app

client = TestClient(app)

VALID_PARAMS = {"route": "DFW-LAX", "model": "sarima", "horizon": 2}

ALL_MODEL_NAMES = ["seasonal_naive", "sarima", "lightgbm", "lstm", "tft"]

def get_forecast(**overrides):
    response = client.get("/forecast", params={**VALID_PARAMS, **overrides})
    return response

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
# ---------- valid requests: each test checks one behavior ----------
def test_forecast_echoes_returns_horizon_points():
    response = get_forecast()
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == VALID_PARAMS["route"]
    assert body["model"] == VALID_PARAMS["model"]
    assert body["horizon"] == VALID_PARAMS["horizon"]
    assert len(body["points"]) == VALID_PARAMS["horizon"] 

def test_forecast_points_are_consistent():
    response = get_forecast()
    assert response.status_code == 200
    body = response.json()
    for point in body["points"]:
        forecast = point["forecast"]
        lower = point["lower"]
        upper = point["upper"]
        assert lower <= forecast
        assert forecast <= upper
        assert lower >=0
        assert forecast >=0
        assert upper >=0

@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_forecast_months_are_first_of_month_and_increasing(horizon):
    response = get_forecast(horizon=horizon)
    assert response.status_code == 200
    body = response.json()
    prev_date = None
    for i, point in enumerate(body["points"]):
        curr_date = date.fromisoformat(point["month"])
        assert curr_date.day == 1
        if len(body["points"]) > 1 and i > 0:
            assert curr_date > prev_date
            month_delta = curr_date.year*12 + curr_date.month - prev_date.year*12 - prev_date.month
            assert month_delta == 1
        prev_date = curr_date


# ---------- parametrized valid cases ----------
@pytest.mark.parametrize("horizon", [1, 12])
def test_forecast_accepts_boundary_horizons(horizon):
    response = get_forecast(horizon=horizon)
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == horizon

@pytest.mark.parametrize("model", ALL_MODEL_NAMES)
def test_forecast_accepts_every_model(model):
    response = get_forecast(model=model)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == model


# ---------- invalid requests: change exactly one thing ----------
@pytest.mark.parametrize("horizon", [0, 13, 99, -1])
def test_forecast_rejects_bad_horizon(horizon):
    response = get_forecast(horizon=horizon)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "horizon"]

def test_forecast_rejects_unknown_model():
    response = get_forecast(model="banana")
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "model"]
    

def test_forecast_rejects_missing_route():
    # the helper can't omit a key by overriding it,
    # so build this request directly from VALID_PARAMS minus "route"
    response = client.get("/forecast", params={k: v for k, v in VALID_PARAMS.items() if k != "route"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "route"]
        