from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

dt_model = joblib.load(MODELS_DIR / "decision_tree.joblib")
lr_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")

@app.get("/")
def root():
    return {"message": "ML Prediction API is running", "status": "healthy"}

class PredictionRequest(BaseModel):
    data: List[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.array(request.data).reshape(1, -1)

    return {
        "decision_tree": int(dt_model.predict(data)[0]),
        "logistic_regression": int(lr_model.predict(data)[0])
    }
