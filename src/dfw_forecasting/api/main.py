from fastapi import FastAPI

app = FastAPI(title="DFW Forecasting API", description="API for monthly passenger demand forecasts for DFW-originating routes", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}