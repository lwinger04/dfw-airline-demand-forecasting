from typing import Literal, Self
from datetime import date
from pydantic import BaseModel, Field, model_validator

ModelTypes = Literal["seasonal_naive", "sarima", "lightgbm", "lstm", "tft"]

class ForecastPoint(BaseModel):
    month: date = Field(description="Month for which the forecast is made")
    forecast: float = Field(ge=0, description="Forecasted demand for the month")
    lower: float = Field(ge=0, description="Lower bound on the demand forecast(95% PI)")
    upper: float = Field(ge=0, description="Upper bound on the demand forecast(95% PI)")
    
    @model_validator(mode='after')
    def check_bounds_valid(self) -> Self:
        if self.forecast < self.lower:
            raise ValueError('Bound error: Forecast is less than lower bound')
        if self.forecast > self.upper:
            raise ValueError('Bound error: Forecast is greater than upper bound')
        return self

class ForecastResponse(BaseModel):
    route: str = Field(description="Route for which the forecast is made")
    model: ModelTypes= Field(description="Model used for forecasting")
    horizon: int = Field(ge=1, le=12, description="Number of months for which the forecast is made")
    points: list[ForecastPoint] = Field(max_length=12, description="List of forecasted demand values for each month in the horizon")
    
    @model_validator(mode='after')
    def check_horizon_valid(self) -> Self:
        if len(self.points) != self.horizon:
            raise ValueError('Horizon error: Number of forecast values does not match the horizon')
        return self