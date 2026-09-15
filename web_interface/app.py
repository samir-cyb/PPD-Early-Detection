"""
PPD Risk Assessment Web Interface
FastAPI backend with real-time SHAP explainability
Model: Gradient Boosting on 17 selected EPDS + PHQ-9 features
Target: EPDS Result (Low / Medium / High)
"""

import os, json, sys
import numpy as np

# ── Pickle compatibility patch ─────────────────────────────────────────────────
# gb_model.pkl was saved in an environment where sklearn's Cython extension
# '_loss' was registered as a top-level module.  Remap it so unpickling works
# regardless of which Python runs the server.
import sklearn._loss._loss as _sklearn_loss
sys.modules.setdefault('_loss', _sklearn_loss)

import joblib
import shap
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PKG_DIR    = os.path.join(BASE_DIR, "..", "shap_package")
MODEL_PATH = os.path.join(PKG_DIR, "gb_model.pkl")
TMPL_PATH  = os.path.join(BASE_DIR, "templates", "index.html")

# ── load model ─────────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)

# ── SHAP explainer ─────────────────────────────────────────────────────────────
# TreeExplainer does NOT support multi-class GradientBoostingClassifier.
# PermutationExplainer works with any model via predict_proba.
# We use a neutral zero-vector background (fast, no training data needed).
_bg = np.zeros((1, 17))   # single neutral baseline sample
explainer = shap.PermutationExplainer(model.predict_proba, _bg)

# ── 17 selected features (exact column names after cleaning) ───────────────────
FEATURES = [
    "You have been so unhappy that you have been crying",
    "You have felt scared or panicky for no good reason",
    "You have felt sad or miserable",
    "Little interest or pleasure in doing things",
    "You have been anxious or worried for no good reason",
    "Feeling down, depressed, or hopeless",
    "Trouble concentrating on things",
    "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
    "Things have been getting to you",
    "Trouble falling or staying asleep, or sleeping too much",
    "You have blamed myself unnecessarily when things went wrong",
    "You have been so unhappy that you have had difficulty sleeping",
    "You have looked forward with enjoyment to things",
    "You have been able to laugh and see the funny side of things",
    "Feeling tired or having little energy",
    "Thoughts that you would be better off dead, or of hurting yourself",
    "The thought of harming yourself has occurred ",   # trailing space matches CSV column
]

# ── LabelEncoder mapping (alphabetical sort of the 4 text categories) ──────────
# Sorted: Around half the days=0, Nearly every day=1, Not at all=2, Several days=3
LABEL_MAP: Dict[str, int] = {
    "Around half the days": 0,
    "Nearly every day":     1,
    "Not at all":           2,
    "Several days":         3,
}

# ── clinical explanations per feature ─────────────────────────────────────────
FEATURE_INFO = {
    "You have been so unhappy that you have been crying": {
        "short": "Crying / tearfulness",
        "scale": "EPDS",
        "meaning": "Persistent crying and tearfulness is one of the most prominent outward signs of postpartum depression. This item strongly distinguishes High-risk mothers from Low and Medium groups.",
    },
    "You have felt scared or panicky for no good reason": {
        "short": "Unexplained fear / panic",
        "scale": "EPDS",
        "meaning": "Postnatal anxiety and panic attacks are tightly linked to PPD. Unexplained fear episodes indicate a heightened stress-response system that raises EPDS risk significantly.",
    },
    "You have felt sad or miserable": {
        "short": "Sadness / misery",
        "scale": "EPDS",
        "meaning": "Core depressed mood — persistent sadness is the central symptom of PPD. Even moderate sadness reported here markedly shifts the risk classification upward.",
    },
    "Little interest or pleasure in doing things": {
        "short": "Loss of interest (anhedonia)",
        "scale": "PHQ-9",
        "meaning": "Anhedonia (inability to feel pleasure) is a key diagnostic marker. Mothers who lose interest in activities they previously enjoyed are at substantially higher PPD risk.",
    },
    "You have been anxious or worried for no good reason": {
        "short": "Unexplained anxiety / worry",
        "scale": "EPDS",
        "meaning": "Excessive worrying without an identifiable cause is characteristic of postnatal anxiety, which co-occurs with PPD in over 60% of cases.",
    },
    "Feeling down, depressed, or hopeless": {
        "short": "Hopelessness / depression",
        "scale": "PHQ-9",
        "meaning": "Persistent hopelessness — the belief that things will not get better — is one of the most clinically serious PPD symptoms and a strong predictor of the High-risk class.",
    },
    "Trouble concentrating on things": {
        "short": "Concentration difficulty",
        "scale": "PHQ-9",
        "meaning": "Cognitive symptoms including poor concentration are common in PPD. Difficulty making decisions or staying focused often reflects underlying depressive neurochemistry.",
    },
    "Feeling bad about yourself or that you are a failure or have let yourself or your family down": {
        "short": "Guilt / self-failure feelings",
        "scale": "PHQ-9",
        "meaning": "Excessive guilt and feelings of failure are hallmarks of clinical depression. For postpartum mothers, these often manifest as feeling like a 'bad mother' — a powerful EPDS risk signal.",
    },
    "Things have been getting to you": {
        "short": "Overwhelm / coping difficulty",
        "scale": "EPDS",
        "meaning": "Feeling overwhelmed and unable to cope with daily demands is a key indicator of PPD severity. It reflects the cumulative emotional load of new motherhood on a vulnerable system.",
    },
    "Trouble falling or staying asleep, or sleeping too much": {
        "short": "Sleep disturbance",
        "scale": "PHQ-9",
        "meaning": "Sleep problems beyond normal newborn-related disruption — either insomnia or hypersomnia — are both symptoms and drivers of worsening PPD.",
    },
    "You have blamed myself unnecessarily when things went wrong": {
        "short": "Excessive self-blame",
        "scale": "EPDS",
        "meaning": "Unnecessary self-blame is an EPDS marker of cognitive distortion. Mothers who chronically blame themselves for minor setbacks show a pattern consistent with depressive thinking.",
    },
    "You have been so unhappy that you have had difficulty sleeping": {
        "short": "Unhappiness-linked insomnia",
        "scale": "EPDS",
        "meaning": "When sleep difficulty is directly caused by unhappiness (rather than infant care demands), it points to depression-driven insomnia — a clinically distinct and more serious pattern.",
    },
    "You have looked forward with enjoyment to things": {
        "short": "Anticipatory pleasure (protective)",
        "scale": "EPDS",
        "meaning": "This reversed item measures positive anticipation. Higher scores (less enjoyment anticipated) push toward High risk. Retaining the ability to look forward to things is a protective signal.",
    },
    "You have been able to laugh and see the funny side of things": {
        "short": "Sense of humour (protective)",
        "scale": "EPDS",
        "meaning": "This reversed item measures emotional resilience. The ability to laugh is one of the first casualties of depression. Low scores (less laughter) strongly push predictions toward High risk.",
    },
    "Feeling tired or having little energy": {
        "short": "Fatigue / low energy",
        "scale": "PHQ-9",
        "meaning": "Persistent fatigue beyond typical newborn care tiredness is a somatic symptom of depression. It contributes moderately to risk but is most significant when combined with mood symptoms.",
    },
    "Thoughts that you would be better off dead, or of hurting yourself": {
        "short": "Passive death ideation ⚠",
        "scale": "PHQ-9",
        "meaning": "Any passive suicidal ideation is a clinical red flag. Even low-frequency occurrence of these thoughts is a serious warning sign that warrants immediate professional evaluation.",
    },
    "The thought of harming yourself has occurred": {
        "short": "Self-harm ideation ⚠",
        "scale": "EPDS",
        "meaning": "Active self-harm ideation is the most severe EPDS item. Any positive response to this item should trigger immediate clinical referral regardless of overall score.",
    },
}

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="PPD Risk Assessment", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class PredictionRequest(BaseModel):
    answers: Dict[str, str]   # { feature_name: "Not at all" | "Several days" | … }


@app.get("/", response_class=HTMLResponse)
async def home():
    with open(TMPL_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/features")
async def get_features():
    """Return feature list + info for the frontend."""
    return {
        "features": FEATURES,
        "options": list(LABEL_MAP.keys()),
        "info": FEATURE_INFO,
    }


@app.post("/predict")
async def predict(req: PredictionRequest):
    # ── build ordered feature vector ──────────────────────────────────────────
    vector = []
    for feat in FEATURES:
        txt = req.answers.get(feat, "Not at all")
        vector.append(LABEL_MAP.get(txt, 2))   # default = "Not at all" = 2

    X = np.array([vector], dtype=float)

    # ── predict ───────────────────────────────────────────────────────────────
    pred_class = str(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    classes    = [str(c) for c in model.classes_]
    proba_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    # ── SHAP values via PermutationExplainer (supports multi-class GB) ───────
    shap_exp  = explainer(X)              # Explanation: values shape (1, 17, 3)
    shap_3d   = shap_exp.values           # (1, 17, n_classes)
    pred_idx  = classes.index(pred_class)
    sv        = shap_3d[0, :, pred_idx]   # shape (17,) for predicted class

    # per-class SHAP for directional arrows
    all_sv = {classes[i]: shap_3d[0, :, i].tolist() for i in range(len(classes))}

    # ── top-3 by |SHAP| for predicted class ──────────────────────────────────
    top3_idx = np.argsort(np.abs(sv))[-3:][::-1]
    top3 = []
    for idx in top3_idx:
        feat = FEATURES[idx]
        info = FEATURE_INFO.get(feat, {})
        enc_val = vector[idx]
        # reverse map encoded → text
        txt_val = next((k for k, v in LABEL_MAP.items() if v == enc_val), "Not at all")
        top3.append({
            "rank":         int(np.where(top3_idx == idx)[0][0] + 1),
            "feature":      feat,
            "short":        info.get("short", feat),
            "scale":        info.get("scale", ""),
            "meaning":      info.get("meaning", ""),
            "shap_value":   round(float(sv[idx]), 5),
            "answer":       txt_val,
            "shap_all_cls": {c: round(float(shap_vals[i][0][idx]), 5) for i, c in enumerate(classes)},
        })

    # ── global shap importance (mean |shap| across all classes) ──────────────
    global_imp = {}
    for i, feat in enumerate(FEATURES):
        mean_abs = float(np.mean(np.abs(shap_3d[0, i, :])))
        global_imp[feat] = round(mean_abs, 5)

    return JSONResponse({
        "prediction":      pred_class,
        "probabilities":   proba_dict,
        "top3_features":   top3,
        "global_importance": global_imp,
        "all_shap":        all_sv,
        "classes":         classes,
        "feature_order":   FEATURES,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
