# End-to-End Machine Learning Application

This project demonstrates a complete **end-to-end Machine Learning workflow**, 
starting from data preparation and model training to backend integration and frontend deployment.

The system uses **Decision Tree** and **Logistic Regression** models trained in a Google Colab–style environment,
exported using `joblib`, served via a **FastAPI backend**, and consumed by a web-based frontend.

---

## Prediction Overview

The application predicts a binary outcome (Approved / Not Approved) based on user-provided numerical inputs:
Age, Monthly Income, Credit Score, and Account Balance.
These features are processed by both models to determine eligibility, allowing comparison between a
linear classification approach (Logistic Regression) and a rule-based non-linear approach (Decision Tree).

---

## 📁 Repositories

### 1️⃣ ML-pipeline (Model Training)


**Responsibilities:**
- Data loading and cleaning
- Model training (Decision Tree & Logistic Regression)
- Model evaluation
- Exporting trained models using `joblib`

---

### 2️⃣ ML-backend (Model Serving)


**Responsibilities:**
- Load trained models
- Expose REST API using FastAPI
- Provide `/predict` endpoint
- Enable frontend integration (CORS)

---

### 3️⃣ ML-frontend (User Interface)


**Responsibilities:**
- Collect user input
- Send requests to backend API
- Display model predictions

---


---

## 🚀 How to Run the Project Locally

### 1️⃣ Train Models (ML-pipeline)
- Open `model_training.ipynb` in Google Colab
- Upload `data.csv`
- Run all cells
- Download `.joblib` files

---

### 2️⃣ Run Backend (ML-backend)
```bash
cd ML-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Backend runs at: http://127.0.0.1:8000


### 3️⃣ Run Frontend (ML-frontend)

- Open index.html in a browser

- Enter feature values

- Click Predict



