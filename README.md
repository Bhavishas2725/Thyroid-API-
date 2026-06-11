# Thyroid Cancer Prediction API

A production-ready FastAPI wrapper around two trained ML models for thyroid cancer outcome prediction. Built with full OpenAPI 3.0 documentation via Swagger UI.

## Results
| Model | Task | Metric |
|---|---|---|
| XGBoost | Recurrence prediction (Yes/No) | High accuracy on clinical dataset |
| Logistic Regression | Risk classification (Low / Intermediate / High) | Trained on same cohort |

## Tech Stack
`Python` · `FastAPI` · `XGBoost` · `Scikit-learn` · `Pandas` · `NumPy` · `Pydantic v2` · `Uvicorn`

---

## Project Structure
```
thyroid-api/
├── main.py              # FastAPI app — routes, schemas, chatbot
├── model_utils.py       # Model loading and preprocessing logic
├── requirements.txt     # All dependencies
├── README.md
│
├── model/               # Place all .pkl files here
│   ├── thyroid_xgb_recur.pkl
│   ├── logistic_risk_model.pkl
│   ├── label_encoders_recur.pkl
│   ├── label_encoders_risk.pkl
│   ├── age_scaler_risk.pkl
│   ├── feature_order_recur.pkl
│   └── feature_order_risk.pkl
│
├── static/css/          # Original CSS (unchanged)
└── templates/           # Original index.html (still served at /ui)
```

---

## Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place your model files**
Copy all `.pkl` files into the `model/` folder.

**3. Run**
```bash
python main.py
```

**4. Open Swagger UI**
```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status + links |
| GET | `/health` | Health check — confirms both models are loaded |
| POST | `/predict` | Recurrence prediction (XGBoost) |
| POST | `/predict_risk` | Risk assessment (Logistic Regression) |
| POST | `/chat` | Clinical assistant chatbot |
| GET | `/ui` | Original HTML form interface |

---

## Example: Recurrence Prediction

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

## Example: Risk Assessment

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

## Disclaimer
This API is a **screening tool only** and is not intended for clinical diagnosis. Always consult a qualified medical professional.

---
