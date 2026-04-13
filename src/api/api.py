from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.models import ClientData
from src.model.predict import Predictor

predictor = Predictor()
app = FastAPI()

@app.get("/", tags=["Health"])
async def check_health():
    return JSONResponse(
        status_code=200,
        content={
            "message" : "ML API is running."
        }
    )

@app.post(
    "/predict", 
    description="Return prediction about term deposit subscription given client data", 
    tags=["ML"]
)
async def predict_deposit(record: ClientData):
    result = predictor.predict(record)          # single dict => 0 or 1
    proba  = predictor.predict_proba(record)    # single dict => float (P(deposit=1))

    return {
        "deposit_prediction": bool(result),
        "deposit_probability": proba
    }
