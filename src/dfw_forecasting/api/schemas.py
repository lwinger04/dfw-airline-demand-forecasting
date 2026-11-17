from datetime import date
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, Field, model_validator

ModelName = Literal["seasonal_naive", "sarima", "lightgbm", "lstm", "tft"]


def verify_first_of_month(v: date) -> date:
    if v.day != 1:
        raise ValueError(f"The date must be the first day of the month. Received: {v}")
    return v


# Reusable type with the after-validator applied
FirstOfMonthDate = Annotated[date, AfterValidator(verify_first_of_month)]


class ForecastPoint(BaseModel):
    month: FirstOfMonthDate = Field(description="Month for which the forecast is made")
    forecast: float = Field(ge=0, description="Forecasted demand for the month")
    lower: float = Field(ge=0, description="Lower bound on the demand forecast(95% PI)")
    upper: float = Field(ge=0, description="Upper bound on the demand forecast(95% PI)")

    @model_validator(mode="after")
    def check_bounds_valid(self) -> Self:
        if self.forecast < self.lower:
            raise ValueError("Bound error: Forecast is less than lower bound")
        if self.forecast > self.upper:
            raise ValueError("Bound error: Forecast is greater than upper bound")
        return self


class ForecastResponse(BaseModel):
    route: str = Field(description="Route for which the forecast is made")
    model: ModelName = Field(description="Model used for forecasting")
    horizon: int = Field(
        ge=1, le=12, description="Number of months for which the forecast is made"
    )
    points: list[ForecastPoint] = Field(
        max_length=12,
        description="List of forecasted demand values for each month in the horizon",
    )

    @model_validator(mode="after")
    def check_horizon_valid(self) -> Self:
        if len(self.points) != self.horizon:
            raise ValueError(
                "Horizon error: Number of forecast values does not match the horizon"
            )
        return self
