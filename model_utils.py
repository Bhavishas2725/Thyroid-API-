"""
model_utils.py
Handles loading of all .pkl artefacts and preprocessing logic
for both the Recurrence and Risk prediction models.
"""
import os
import joblib
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def _pkl(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_all():
    """
    Load every artefact needed by both prediction endpoints.
    Returns a dict so main.py can store them in app.state.
    """
    artefacts = {}

    required = {
        "recur_model":    "thyroid_xgb_recur.pkl",
        "risk_model":     "logistic_risk_model.pkl",
        "le_recur":       "label_encoders_recur.pkl",
        "le_risk":        "label_encoders_risk.pkl",
        "scaler":         "age_scaler_risk.pkl",
        "feat_recur":     "feature_order_recur.pkl",
        "feat_risk":      "feature_order_risk.pkl",
    }

    for key, filename in required.items():
        path = _pkl(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model file not found: model/{filename}\n"
                f"Please place all .pkl files inside the model/ folder."
            )
        artefacts[key] = joblib.load(path)
        print(f"  [OK] Loaded {filename}")

    return artefacts


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(raw: dict, label_encoders: dict, scaler, feature_order: list) -> pd.DataFrame:
    """
    Encode categoricals, scale Age, and select/order features
    exactly as the model expects.
    """
    df = pd.DataFrame([raw])

    # Encode each categorical column using its fitted LabelEncoder
    for col, enc in label_encoders.items():
        if col in df.columns and hasattr(enc, "classes_"):
            mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # Scale Age
    if scaler is not None and hasattr(scaler, "transform"):
        df["Age"] = scaler.transform(df[["Age"]].values).flatten()

    # Select only the features the model was trained on, in the right order
    cols = [c for c in feature_order if c in df.columns]
    df = df[cols]

    # Safety: coerce any remaining non-numeric to -1
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    return df


# ── Prediction helpers ───────────────────────────────────────────────────────
def run_recurrence(artefacts: dict, raw: dict) -> dict:
    df = preprocess(
        raw,
        label_encoders=artefacts["le_recur"],
        scaler=artefacts["scaler"],
        feature_order=list(artefacts["feat_recur"]),
    )
    model = artefacts["recur_model"]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    label = "Yes" if pred == 1 else "No"
    confidence = round(float(max(proba)) * 100, 2)

    return {"recurrence": label, "confidence_pct": confidence}


def run_risk(artefacts: dict, raw: dict) -> dict:
    df = preprocess(
        raw,
        label_encoders=artefacts["le_risk"],
        scaler=artefacts["scaler"],
        feature_order=list(artefacts["feat_risk"]),
    )
    model = artefacts["risk_model"]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    risk_map = {0: "Low", 1: "High", 2: "Intermediate"}
    label = risk_map.get(int(pred), str(pred))
    confidence = round(float(max(proba)) * 100, 2)

    return {"risk_level": label, "confidence_pct": confidence}
