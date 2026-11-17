# ---------- imports ----------
from datetime import date

import pytest
from pydantic import ValidationError

from dfw_forecasting.api.schemas import ForecastPoint, ForecastResponse


# ---------- helpers ----------
def valid_point_kwargs(**overrides):
    valid_point = {
        "month": date(2026, 2, 1),
        "forecast": 100.0,
        "lower": 80.0,
        "upper": 120.0,
    }
    valid_point.update(overrides)
    return valid_point


def valid_response_kwargs(n_points=2, **overrides):
    response = {
        "route": "DFW-LAX",
        "model": "sarima",
        "horizon": n_points,
        "points": [
            ForecastPoint(**valid_point_kwargs(month=date(2026, 2 + i, 1)))
            for i in range(n_points)
        ],
    }
    response.update(overrides)
    return response


# ---------- ForecastPoint: happy paths ----------
def test_point_accepts_valid_values():
    point = ForecastPoint(**valid_point_kwargs())
    check = valid_point_kwargs()
    assert point.month == check["month"]
    assert point.forecast == check["forecast"]
    assert point.lower == check["lower"]
    assert point.upper == check["upper"]


def test_point_accepts_equal_bounds():
    point = ForecastPoint(
        **valid_point_kwargs(lower=100.0, forecast=100.0, upper=100.0)
    )
    assert point.lower == point.forecast == point.upper


def test_point_accepts_first_of_month():
    point = ForecastPoint(**valid_point_kwargs(month=date(2026, 10, 1)))
    assert point.month == date(2026, 10, 1)


# ---------- ForecastPoint: unhappy paths ----------
def test_point_rejects_lower_above_forecast():
    with pytest.raises(ValidationError, match="Forecast is less than lower bound"):
        ForecastPoint(**valid_point_kwargs(lower=150.0, forecast=100.0, upper=200.0))


def test_point_rejects_upper_below_forecast():
    with pytest.raises(ValidationError, match="Forecast is greater than upper bound"):
        ForecastPoint(**valid_point_kwargs(lower=50.0, forecast=150.0, upper=100.0))


@pytest.mark.parametrize("field", ["forecast", "lower", "upper"])
def test_point_rejects_negative_values(field):
    with pytest.raises(ValidationError) as excinfo:
        ForecastPoint(**valid_point_kwargs(**{field: -1.0}))
    error = excinfo.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == "greater_than_equal"


def test_point_rejects_month_not_on_first():
    with pytest.raises(ValidationError) as excinfo:
        ForecastPoint(**valid_point_kwargs(month=date(2026, 2, 17)))
    assert excinfo.value.errors()[0]["loc"] == ("month",)


def test_response_rejects_too_few_points():
    with pytest.raises(
        ValidationError, match="Number of forecast values does not match the horizon"
    ):
        ForecastResponse(**valid_response_kwargs(n_points=2, horizon=3))


def test_response_rejects_too_many_points():
    # more points than horizon; keep point count <= 12 so max_length isn't what fires
    with pytest.raises(
        ValidationError, match="Number of forecast values does not match the horizon"
    ):
        ForecastResponse(**valid_response_kwargs(n_points=3, horizon=2))


@pytest.mark.parametrize(
    ("horizon", "expected_type"), [(0, "greater_than_equal"), (13, "less_than_equal")]
)
def test_response_rejects_out_of_range_horizon(horizon, expected_type):
    with pytest.raises(ValidationError) as excinfo:
        ForecastResponse(**valid_response_kwargs(horizon=horizon))
    assert excinfo.value.errors()[0]["loc"] == ("horizon",)
    assert excinfo.value.errors()[0]["type"] == expected_type


def test_response_rejects_unknown_model():
    with pytest.raises(ValidationError) as excinfo:
        ForecastResponse(**valid_response_kwargs(model="banana"))
    assert excinfo.value.errors()[0]["loc"] == ("model",)
    assert excinfo.value.errors()[0]["type"] == "literal_error"
