from dfw_forecasting.api.schemas import ForecastResponse, ForecastPoint, ModelName
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Annotated
from fastapi import FastAPI, Query

app = FastAPI(title="DFW Forecasting API", description="API for monthly passenger demand forecasts for DFW-originating routes", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast")
def forecast(route: str, model: ModelName, horizon: Annotated[int, Query(ge=1, le=12)]) -> ForecastResponse:
    last_data_month = date(2026,1,1)
    forecasted_months = [last_data_month + relativedelta(months=i) for i in range(1, horizon + 1)]
    forecast_points = [ForecastPoint(month=month, forecast=10000.0*i, lower=5000.0*i, upper=15000.0*i) for i, month in enumerate(forecasted_months,start=1)]
    return ForecastResponse(route=route, model=model, horizon=horizon, points=forecast_points)