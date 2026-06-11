"""
main.py
Thyroid Cancer Prediction API — FastAPI + Swagger UI
Endpoints:
  GET  /             → status
  GET  /health       → health check
  POST /predict      → recurrence prediction  (XGBoost)
  POST /predict_risk → risk assessment        (Logistic Regression)
  POST /chat         → clinical chatbot
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn
import os

from model_utils import load_all, run_recurrence, run_risk

# ── Lifespan: load models once at startup ────────────────────────────────────
artefacts: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nLoading model artefacts...")
    artefacts.update(load_all())
    print("All models ready.\n")
    yield
    artefacts.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Thyroid Cancer Prediction API",
    description="""
## Thyroid Cancer Prediction API

A production-ready REST API for thyroid cancer outcome prediction using
two trained machine learning models:

| Endpoint | Model | Output |
|---|---|---|
| `/predict` | XGBoost | Recurrence — Yes / No |
| `/predict_risk` | Logistic Regression | Risk Level — Low / Intermediate / High |

### How to use
1. Expand an endpoint below
2. Click **Try it out**
3. Fill in the patient details and click **Execute**

> ⚠️ This is a **clinical screening tool only**, not a substitute for medical diagnosis.
    """,
    version="1.0.0",
    contact={
        "name": "Thyroid Prediction Project",
        "url": "https://github.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files + templates (for the original HTML UI)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates") if os.path.exists("templates") else None


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

# Shared fields across both endpoints
class _SharedPatientFields(BaseModel):
    age: int = Field(..., ge=1, le=120, example=45,
                     description="Patient age in years")
    gender: Literal["F", "M"] = Field(..., example="F",
                                      description="Patient gender")
    smoking: Literal["Yes", "No"] = Field(..., example="No",
                                          description="Current smoker?")
    hx_smoking: Literal["Yes", "No"] = Field(..., example="No",
                                              description="History of smoking")
    hx_radiotherapy: Literal["Yes", "No"] = Field(..., example="No",
                                                   description="History of radiotherapy")
    thyroid_function: Literal[
        "Euthyroid",
        "Clinical Hyperthyroidism",
        "Clinical Hypothyroidism",
        "Subclinical Hyperthyroidism",
        "Subclinical Hypothyroidism",
    ] = Field(..., example="Euthyroid", description="Thyroid function status")
    physical_examination: Literal[
        "Diffuse goiter",
        "Multinodular goiter",
        "Normal",
        "Single nodular goiter-left",
        "Single nodular goiter-right",
    ] = Field(..., example="Single nodular goiter-left",
              description="Physical examination finding")
    adenopathy: Literal[
        "No", "Right", "Left", "Bilateral", "Extensive", "Posterior"
    ] = Field(..., example="No", description="Adenopathy presence and location")
    pathology: Literal[
        "Papillary", "Micropapillary", "Follicular", "Hurthel cell"
    ] = Field(..., example="Papillary", description="Thyroid pathology type")
    focality: Literal["Uni-Focal", "Multi-Focal"] = Field(..., example="Uni-Focal",
                                                          description="Tumour focality")
    stage: Literal["I", "II", "III", "IVA", "IVB"] = Field(..., example="I",
                                                            description="Overall cancer stage")


class RecurrenceInput(_SharedPatientFields):
    """Input schema for recurrence prediction.
    Requires all shared fields plus `risk` and `response`."""
    risk: Literal["Low", "High"] = Field(..., example="Low",
                                         description="Assigned risk level")
    response: Literal[
        "Excellent", "Biochemical Incomplete", "Indeterminate", "Structural Incomplete"
    ] = Field(..., example="Excellent", description="Treatment response")


class RiskInput(_SharedPatientFields):
    """Input schema for risk assessment.
    Uses TNM staging instead of risk/response fields."""
    pass   # Risk model only uses the 11 shared fields (no T/N/M in feature_order_risk)


class RecurrenceResponse(BaseModel):
    recurrence: str = Field(..., example="No",
                            description="Whether recurrence is predicted: Yes or No")
    confidence_pct: float = Field(..., example=87.43,
                                  description="Model confidence as a percentage")
    message: str = Field(..., example="Recurrence Prediction: No (87.43% confidence)")

class RiskResponse(BaseModel):
    risk_level: str = Field(..., example="Low",
                            description="Predicted risk level: Low, Intermediate, or High")
    confidence_pct: float = Field(..., example=91.2,
                                  description="Model confidence as a percentage")
    message: str = Field(..., example="Risk Level: Low (91.2% confidence)")

class ChatInput(BaseModel):
    message: str = Field(..., example="What is recurrence prediction?",
                         description="Your question for the thyroid assistant")

class ChatResponse(BaseModel):
    response: str = Field(..., example="Recurrence prediction determines if cancer is likely to return.")

class HealthResponse(BaseModel):
    status: str
    recurrence_model_loaded: bool
    risk_model_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"], summary="API root")
def root():
    """Returns a welcome message and links to documentation."""
    return {
        "message": "Thyroid Cancer Prediction API is running ✓",
        "swagger_ui": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Status"],
         summary="Health check")
def health():
    """Confirms the API is live and both models are loaded."""
    return HealthResponse(
        status="healthy" if artefacts else "degraded",
        recurrence_model_loaded="recur_model" in artefacts,
        risk_model_loaded="risk_model" in artefacts,
    )


@app.post(
    "/predict",
    response_model=RecurrenceResponse,
    tags=["Predictions"],
    summary="Predict cancer recurrence",
    responses={
        200: {"description": "Recurrence prediction returned successfully"},
        422: {"description": "Validation error — check field values"},
        500: {"description": "Model inference error"},
    },
)
def predict_recurrence(data: RecurrenceInput):
    """
    **Predict whether thyroid cancer is likely to recur.**

    Accepts 13 clinical features and returns `Yes` or `No` along with
    the model's confidence score. Powered by a trained **XGBoost** classifier.

    ### Required fields
    - Core patient info: age, gender, smoking history, radiotherapy history
    - Clinical findings: thyroid function, physical examination, adenopathy, pathology, focality
    - Staging & outcome: stage, risk level, treatment response
    """
    if not artefacts:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Try again shortly.")
    try:
        raw = {
            "Age":                  data.age,
            "Gender":               data.gender,
            "Smoking":              data.smoking,
            "Hx Smoking":           data.hx_smoking,
            "Hx Radiothreapy":      data.hx_radiotherapy,
            "Thyroid Function":     data.thyroid_function,
            "Physical Examination": data.physical_examination,
            "Adenopathy":           data.adenopathy,
            "Pathology":            data.pathology,
            "Focality":             data.focality,
            "Stage":                data.stage,
            "Risk":                 data.risk,
            "Response":             data.response,
        }
        result = run_recurrence(artefacts, raw)
        return RecurrenceResponse(
            recurrence=result["recurrence"],
            confidence_pct=result["confidence_pct"],
            message=f"Recurrence Prediction: {result['recurrence']} "
                    f"({result['confidence_pct']}% confidence)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict_risk",
    response_model=RiskResponse,
    tags=["Predictions"],
    summary="Assess patient risk level",
    responses={
        200: {"description": "Risk assessment returned successfully"},
        422: {"description": "Validation error — check field values"},
        500: {"description": "Model inference error"},
    },
)
def predict_risk(data: RiskInput):
    """
    **Classify patient into Low, Intermediate, or High risk.**

    Accepts 11 clinical features and returns a risk category along with
    the model's confidence score. Powered by a trained **Logistic Regression** model.

    ### Required fields
    - Core patient info: age, gender, smoking history, radiotherapy history
    - Clinical findings: thyroid function, physical examination, adenopathy, pathology, focality
    - Overall stage
    """
    if not artefacts:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Try again shortly.")
    try:
        raw = {
            "Age":                  data.age,
            "Gender":               data.gender,
            "Smoking":              data.smoking,
            "Hx Smoking":           data.hx_smoking,
            "Hx Radiothreapy":      data.hx_radiotherapy,
            "Thyroid Function":     data.thyroid_function,
            "Physical Examination": data.physical_examination,
            "Adenopathy":           data.adenopathy,
            "Pathology":            data.pathology,
            "Focality":             data.focality,
            "Stage":                data.stage,
        }
        result = run_risk(artefacts, raw)
        return RiskResponse(
            risk_level=result["risk_level"],
            confidence_pct=result["confidence_pct"],
            message=f"Risk Level: {result['risk_level']} "
                    f"({result['confidence_pct']}% confidence)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {str(e)}")


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chatbot"],
    summary="Clinical assistant chatbot",
)
def chat(data: ChatInput):
    """
    **Rule-based clinical assistant.**

    Ask questions about the API, thyroid cancer, risk assessment,
    or recurrence prediction.

    **Example questions:**
    - *"What is recurrence prediction?"*
    - *"How do I use this API?"*
    - *"What does risk level mean?"*
    """
    reply = _chatbot(data.message.lower())
    return ChatResponse(response=reply)


def _chatbot(msg: str) -> str:
    if any(w in msg for w in ["hello", "hi", "hey"]):
        return "Hello! I'm your Thyroid Assistant. Ask me about recurrence prediction, risk assessment, or how to use this API."
    if any(w in msg for w in ["how", "use", "help"]):
        return "POST patient data to /predict for recurrence or /predict_risk for risk assessment. Full docs at /docs."
    if any(w in msg for w in ["risk", "risk level", "assessment"]):
        return "Risk assessment classifies patients as Low, Intermediate, or High risk using a Logistic Regression model trained on clinical features."
    if any(w in msg for w in ["recurrence", "recur"]):
        return "Recurrence prediction uses XGBoost to determine if thyroid cancer is likely to return (Yes/No) after treatment."
    if any(w in msg for w in ["cancer", "thyroid"]):
        return "Thyroid cancer forms in thyroid gland tissue. Early risk stratification helps guide treatment decisions."
    if any(w in msg for w in ["model", "algorithm", "accuracy"]):
        return "This API uses XGBoost for recurrence (high accuracy on clinical data) and Logistic Regression for risk stratification."
    if any(w in msg for w in ["confidence", "probability"]):
        return "Confidence % reflects how certain the model is about its prediction. Higher is more certain, but always consult a clinician."
    return "I'm not sure about that. Try asking about recurrence prediction, risk levels, or how to use the API. Full docs at /docs."


# ── HTML UI (legacy Flask template still served) ──────────────────────────────
@app.get("/ui", response_class=HTMLResponse, tags=["UI"],
         summary="Original HTML prediction interface",
         include_in_schema=True)
def serve_ui(request: Request):
    """Serves the original HTML form-based interface."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": ""})
    return HTMLResponse("<h2>HTML template not found. Use the <a href='/docs'>API docs</a> instead.</h2>")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
