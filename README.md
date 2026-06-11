#  Thyroid Cancer Prediction API

> A **production-ready REST API** built with **FastAPI** that wraps two trained ML models for thyroid cancer outcome prediction — featuring full **OpenAPI 3.0 / Swagger UI** documentation, Pydantic v2 validation, confidence scores, and a legacy HTML interface.

> 🔗 This project is a **v2 upgrade** of the [Thyroid Classification System](https://github.com/Bhavishas2725/Thyroid-Prediction) — the same models and preprocessing pipeline, now exposed as a clean, documented REST API.

---

##  Table of Contents

- [Overview](#overview)
- [What's New in v2](#whats-new-in-v2)
- [Features](#features)
- [Models & Results](#models--results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Request & Response Examples](#request--response-examples)
- [Swagger UI](#swagger-ui)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Overview

The **Thyroid Cancer Prediction API** exposes two machine learning models as REST endpoints for seamless integration into clinical dashboards, frontend apps, or third-party systems:

| Endpoint | Model | Output |
|---|---|---|
| `POST /predict` | XGBoost | Recurrence — `Yes` / `No` + confidence % |
| `POST /predict_risk` | Logistic Regression | Risk Level — `Low` / `Intermediate` / `High` + confidence % |

Both endpoints return a **confidence percentage** alongside the prediction — a key improvement over the original Flask interface.

---

## What's New in v2

| Feature | Flask v1 | FastAPI v2 |
|---|---|---|
| Framework | Flask 2.0 | **FastAPI 0.111** |
| API Documentation | ❌ None | ✅ **Swagger UI + ReDoc** |
| Input Validation | Manual | ✅ **Pydantic v2 schemas** |
| Confidence Score | ❌ | ✅ `confidence_pct` in every response |
| Health Check Endpoint | ❌ | ✅ `/health` |
| CORS Support | ❌ | ✅ Enabled for all origins |
| Async Lifespan (model loading) | ❌ | ✅ Models loaded once at startup |
| Legacy HTML UI | ✅ Main interface | ✅ Still available at `/ui` |
| `curl` / HTTP client ready | ❌ | ✅ Full JSON API |

---

## Features

- **FastAPI** with async lifespan — models are loaded **once at startup**, not per request
- **Auto-generated Swagger UI** at `/docs` and ReDoc at `/redoc`
- **Pydantic v2 validation** — strict `Literal` types for all categorical fields prevent invalid input
- **Confidence scores** — every prediction includes the model's probability as a percentage
- **Health check endpoint** — confirms both models are loaded and ready
- **CORS middleware** — open for cross-origin requests (configurable)
- **Legacy HTML UI** — original form-based interface still served at `/ui`
- **Clinical chatbot** — rule-based assistant available at `POST /chat`
- **Clean preprocessing module** — `model_utils.py` separates all loading and inference logic from routing

---

## Models & Results

> Both models are trained on the same dataset and preprocessing pipeline from v1. Model files are shared between the Flask and FastAPI projects.

### Recurrence Prediction — XGBoost Classifier

**XGBoost Hyperparameters:**

```
n_estimators     = 500
learning_rate    = 0.03
max_depth        = 4
min_child_weight = 1
subsample        = 0.8
colsample_bytree = 0.8
gamma            = 0.1
reg_alpha        = 0.5
objective        = binary:logistic
eval_metric      = logloss
random_state     = 42
```

**Classification Report (Test Set — 80/20 Split):**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Recurrence (0) | 0.98 | 0.92 | **0.95** | 51 |
| Recurrence (1) | 0.84 | 0.95 | **0.89** | 22 |
| **Macro Avg** | 0.91 | 0.94 | **0.92** | 73 |
| **Weighted Avg** | 0.94 | 0.93 | **0.93** | 73 |

**Overall Accuracy: 93.15%**

---

### Risk Assessment — Logistic Regression

```
penalty      = l2
C            = 1.0
solver       = liblinear
class_weight = balanced
max_iter     = 1000
```

**Overall Accuracy: 82.19%**

---

## Project Structure

```
thyroid-api/
│
├── main.py                         # FastAPI app — routes, Pydantic schemas, chatbot
├── model_utils.py                  # Model loading, preprocessing & prediction helpers
├── requirements.txt                # Pinned Python dependencies
├── README.md                       # Project documentation
├── .gitignore
│
├── model/                          # Pre-trained model artefacts (place .pkl files here)
│   ├── thyroid_xgb_recur.pkl       # XGBoost — recurrence model
│   ├── logistic_risk_model.pkl     # Logistic Regression — risk model
│   ├── label_encoders_recur.pkl    # Label encoders (recurrence pipeline)
│   ├── label_encoders_risk.pkl     # Label encoders (risk pipeline)
│   ├── age_scaler_risk.pkl         # StandardScaler for age feature
│   ├── feature_order_recur.pkl     # Feature order (recurrence model)
│   └── feature_order_risk.pkl      # Feature order (risk model)
│
├── templates/
│   └── index.html                  # Legacy HTML prediction form (served at /ui)
│
└── static/
    └── css/
        └── style.css               # Glassmorphism UI styling
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/thyroid-cancer-api.git
cd thyroid-cancer-api

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model files
# Copy all .pkl files into the model/ directory
```

---

## Running the API

```bash
python main.py
```

The server starts at:

```
http://localhost:8000
```

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | ✅ Swagger UI (interactive API docs) |
| `http://localhost:8000/redoc` | ✅ ReDoc documentation |
| `http://localhost:8000/health` | ✅ Health check |
| `http://localhost:8000/ui` | ✅ Legacy HTML form interface |

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/` | API status + documentation links | None |
| `GET` | `/health` | Health check — confirms both models are loaded | None |
| `POST` | `/predict` | Recurrence prediction (XGBoost) | None |
| `POST` | `/predict_risk` | Risk level classification (Logistic Regression) | None |
| `POST` | `/chat` | Clinical assistant chatbot | None |
| `GET` | `/ui` | Original HTML form interface | None |

---

## Request & Response Examples

### POST `/predict` — Recurrence Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "F",
    "smoking": "No",
    "hx_smoking": "No",
    "hx_radiotherapy": "No",
    "thyroid_function": "Euthyroid",
    "physical_examination": "Single nodular goiter-left",
    "adenopathy": "No",
    "pathology": "Papillary",
    "focality": "Uni-Focal",
    "stage": "I",
    "risk": "Low",
    "response": "Excellent"
  }'
```

**Response:**
```json
{
  "recurrence": "No",
  "confidence_pct": 91.24,
  "message": "Recurrence Prediction: No (91.24% confidence)"
}
```

---

### POST `/predict_risk` — Risk Assessment

**Request:**
```bash
curl -X POST "http://localhost:8000/predict_risk" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "gender": "M",
    "smoking": "Yes",
    "hx_smoking": "Yes",
    "hx_radiotherapy": "No",
    "thyroid_function": "Euthyroid",
    "physical_examination": "Multinodular goiter",
    "adenopathy": "Right",
    "pathology": "Follicular",
    "focality": "Multi-Focal",
    "stage": "III"
  }'
```

**Response:**
```json
{
  "risk_level": "High",
  "confidence_pct": 88.5,
  "message": "Risk Level: High (88.5% confidence)"
}
```

---

### GET `/health` — Health Check

**Response:**
```json
{
  "status": "healthy",
  "recurrence_model_loaded": true,
  "risk_model_loaded": true
}
```

---

## Input Field Reference

### Recurrence Prediction (`/predict`) — 13 Fields

| Field | Type | Accepted Values |
|---|---|---|
| `age` | int | 1 – 120 |
| `gender` | string | `"F"`, `"M"` |
| `smoking` | string | `"Yes"`, `"No"` |
| `hx_smoking` | string | `"Yes"`, `"No"` |
| `hx_radiotherapy` | string | `"Yes"`, `"No"` |
| `thyroid_function` | string | `"Euthyroid"`, `"Clinical Hyperthyroidism"`, `"Clinical Hypothyroidism"`, `"Subclinical Hyperthyroidism"`, `"Subclinical Hypothyroidism"` |
| `physical_examination` | string | `"Diffuse goiter"`, `"Multinodular goiter"`, `"Normal"`, `"Single nodular goiter-left"`, `"Single nodular goiter-right"` |
| `adenopathy` | string | `"No"`, `"Right"`, `"Left"`, `"Bilateral"`, `"Extensive"`, `"Posterior"` |
| `pathology` | string | `"Papillary"`, `"Micropapillary"`, `"Follicular"`, `"Hurthel cell"` |
| `focality` | string | `"Uni-Focal"`, `"Multi-Focal"` |
| `stage` | string | `"I"`, `"II"`, `"III"`, `"IVA"`, `"IVB"` |
| `risk` | string | `"Low"`, `"High"` |
| `response` | string | `"Excellent"`, `"Biochemical Incomplete"`, `"Indeterminate"`, `"Structural Incomplete"` |

### Risk Assessment (`/predict_risk`) — 11 Fields

Same as above but **without** `risk` and `response` fields.

---

## Swagger UI

The API ships with full interactive documentation powered by **Swagger UI**. Navigate to `http://localhost:8000/docs` to:

- Browse all endpoints and their schemas
- Try out live predictions directly in the browser
- Inspect request/response models and field constraints
- View all accepted `Literal` values per field

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **API Framework** | FastAPI | 0.111.0 |
| **Server** | Uvicorn | 0.29.0 |
| **Data Validation** | Pydantic | 2.7.1 |
| **ML — Recurrence** | XGBoost | 2.0.3 |
| **ML — Risk** | Scikit-learn (Logistic Regression) | 1.4.2 |
| **Data Processing** | Pandas | 2.2.2 |
| **Numerical** | NumPy | 1.26.4 |
| **Model Persistence** | Joblib | 1.4.2 |
| **Templating** | Jinja2 | 3.1.4 |
| **Language** | Python | 3.9+ |

---

## Related Project

> 🔗 **[Thyroid Classification System (Flask v1)](https://github.com/Bhavishas2725/Thyroid-Prediction)** — The original web application with an HTML form interface. This API project wraps the same trained models and preprocessing pipeline.

---

## Disclaimer

> ⚠️ This API is a **clinical screening tool only** and is **not** intended as a substitute for professional medical diagnosis or clinical judgment. Always consult a qualified medical professional for diagnosis and treatment decisions.

---

## License

This project is licensed under the [MIT License](LICENSE).
