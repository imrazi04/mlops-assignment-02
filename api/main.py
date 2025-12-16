from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from typing import List

app = FastAPI()

MODEL_PATH = "models/model.pkl"

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(req: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        return {"error": "model not found"}
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X = [req.features]
    pred = model.predict(X).tolist()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()
    else:
        proba = None
    return {"prediction": pred, "probabilities": proba}
