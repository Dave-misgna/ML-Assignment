from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Frontend is in ML-frontend directory (when root is ML-app)
# Go up from app/ to ML-backend/ to ML-app/, then into ML-frontend/
FRONTEND_HTML = BASE_DIR.parent.parent / "ML-frontend" / "index.html"

# Initialize models as None
dt_model = None
lr_model = None

# Load models with error handling
try:
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Models directory exists: {MODELS_DIR.exists()}")
    
    if MODELS_DIR.exists():
        logger.info(f"Files in models directory: {list(MODELS_DIR.iterdir())}")
    
    dt_path = MODELS_DIR / "decision_tree.joblib"
    lr_path = MODELS_DIR / "logistic_regression.joblib"
    
    logger.info(f"Loading decision tree from: {dt_path}")
    logger.info(f"Decision tree file exists: {dt_path.exists()}")
    dt_model = joblib.load(dt_path)
    logger.info("Decision tree model loaded successfully")
    
    logger.info(f"Loading logistic regression from: {lr_path}")
    logger.info(f"Logistic regression file exists: {lr_path.exists()}")
    lr_model = joblib.load(lr_path)
    logger.info("Logistic regression model loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}", exc_info=True)
    sys.exit(1)

@app.get("/", response_class=HTMLResponse)
def root():
    # Serve the frontend HTML file
    if FRONTEND_HTML.exists():
        with open(FRONTEND_HTML, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return f"""
        <html>
            <body>
                <h1>ML Prediction API</h1>
                <p>Status: healthy</p>
                <p>Models loaded: {dt_model is not None and lr_model is not None}</p>
                <p>Frontend file not found at: {FRONTEND_HTML}</p>
            </body>
        </html>
        """

class PredictionRequest(BaseModel):
    data: List[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    if dt_model is None or lr_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    data = np.array(request.data).reshape(1, -1)

    return {
        "decision_tree": int(dt_model.predict(data)[0]),
        "logistic_regression": int(lr_model.predict(data)[0])
    }
