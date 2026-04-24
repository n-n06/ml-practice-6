from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.schemas import ClientData
from src.model.predict import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield

app = FastAPI(lifespan=lifespan)

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
async def predict_deposit(record: ClientData, request: Request):
    predictor = request.app.state.predictor

    result = predictor.predict(record)          # single dict => 0 or 1
    proba  = predictor.predict_proba(record)    # single dict => float (P(deposit=1))

    return {
        "deposit_prediction": bool(result),
        "deposit_probability": proba
    }
