"""
PPD Risk Assessment Web Interface
FastAPI backend with real-time SHAP explainability
Model: Logistic Regression (multinomial) on 16 selected EPDS + PHQ-9 features
Target: EPDS Result (Low / Medium / High)
Encoding: Ordinal (Not at all=0, Several days=1, Around half=2, Nearly every day=3)
SHAP: LinearExplainer
"""

import os, json, sys, logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)

import numpy as np
import joblib
import shap
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PKG_DIR     = os.path.join(BASE_DIR, "..", "shap_lr_package")
MODEL_PATH  = os.path.join(PKG_DIR, "lr_model.pkl")
SCALER_PATH = os.path.join(PKG_DIR, "lr_scaler.pkl")
TMPL_PATH   = os.path.join(BASE_DIR, "templates", "index.html")

# ── load model + scaler ────────────────────────────────────────────────────────
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Decode LabelEncoded class integers back to EPDS Result names ───────────────
# EPDS Result encoded alphabetically before training:
#   0 → "High"   (H < L < M)
#   1 → "Low"
#   2 → "Medium"
CLASS_DECODE = {"0": "High", "1": "Low", "2": "Medium"}

# ── SHAP explainer (LinearExplainer — exact for linear models) ─────────────────
# LinearExplainer is the correct and fast explainer for Logistic Regression.
# Background: scaled zero-vector (neutral baseline).
_bg = scaler.transform(np.zeros((1, 16)))
explainer = shap.LinearExplainer(model, shap.maskers.Independent(_bg))

# ── Print model classes at startup ────────────────────────────────────────────
import logging as _lg
_log = _lg.getLogger("ppd.startup")
_log.warning("Model type: %s", type(model).__name__)
_log.warning("Model classes: %s", model.classes_)
_log.warning("CLASS_DECODE: %s", CLASS_DECODE)

# ── 16 selected features — EXACT TRAINING COLUMN ORDER ────────────────────────
# Source: chi2_optimal_features.txt — this is the order features appear in the
# training matrix X = df[optimal_features].copy() used to fit lr_model.pkl.
# Order comes from the chi2 backward-elimination history list, NOT from the
# chi2 importance ranking.  Preserving this order is critical: the scaler and
# model.coef_ are both indexed against THIS ordering.
#
# Key positions:
#   [0]  hopelessness    coef_High = +1.760  (positive → pushes High)
#   [7]  crying          coef_High = +2.137  (positive → pushes High)
#   [8]  laugh           coef_High = -1.732  (REVERSED: more laugh = less High)
#   [9]  look_forward    coef_High = -1.732  (REVERSED: more anticipation = less High)
#   [15] sleep           coef_High = +2.318  (positive → pushes High, strongest feature)
#
# "You have been so unhappy that you have had difficulty sleeping" was removed
# by backward elimination when correct ordinal encoding was applied.
FEATURES = [
    "Feeling down, depressed, or hopeless",                                                      # [0]
    "You have felt sad or miserable",                                                            # [1]
    "You have been anxious or worried for no good reason",                                       # [2]
    "Trouble concentrating on things",                                                           # [3]
    "Things have been getting to you",                                                           # [4]
    "Feeling bad about yourself or that you are a failure or have let yourself or your family down", # [5]
    "You have blamed myself unnecessarily when things went wrong",                               # [6]
    "You have been so unhappy that you have been crying",                                        # [7]
    "You have been able to laugh and see the funny side of things",                              # [8]  ← reversed item
    "You have looked forward with enjoyment to things",                                          # [9]  ← reversed item
    "You have felt scared or panicky for no good reason",                                        # [10]
    "Little interest or pleasure in doing things",                                               # [11]
    "Feeling tired or having little energy",                                                     # [12]
    "Thoughts that you would be better off dead, or of hurting yourself",                        # [13]
    "The thought of harming yourself has occurred",                                              # [14]
    "Trouble falling or staying asleep, or sleeping too much",                                   # [15]
]

# ── Answer encoding (correct ordinal — clinical severity order) ────────────────
# Retraining with ordinal encoding (not alphabetical LabelEncoder):
#   Not at all=0 (least severe) → Nearly every day=3 (most severe)
LABEL_MAP: Dict[str, int] = {
    "Not at all":           0,
    "Several days":         1,
    "Around half the days": 2,
    "Nearly every day":     3,
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
        "meaning": "Sleep problems beyond normal newborn-related disruption — either insomnia or hypersomnia — are the single strongest SHAP predictor for this model. They are both a symptom and a driver of worsening PPD.",
    },
    "You have blamed myself unnecessarily when things went wrong": {
        "short": "Excessive self-blame",
        "scale": "EPDS",
        "meaning": "Unnecessary self-blame is an EPDS marker of cognitive distortion. Mothers who chronically blame themselves for minor setbacks show a pattern consistent with depressive thinking.",
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
app = FastAPI(title="PPD Risk Assessment", version="2.0")
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


@app.get("/debug")
async def debug():
    """
    Diagnostic endpoint — visit http://localhost:8080/debug in browser.
    Shows model internals and runs sanity-check predictions.
    """
    def _pred(raw_arr):
        sc = scaler.transform(raw_arr)
        raw = str(model.predict(sc)[0])
        cls = CLASS_DECODE.get(raw, raw)
        prob = {CLASS_DECODE.get(str(c), str(c)): round(float(p), 4)
                for c, p in zip(model.classes_, model.predict_proba(sc)[0])}
        return cls, prob

    # Sanity checks: ordinal encoding — Nearly every day=3 should be High
    cls_nearly,  proba_nearly    = _pred(np.full((1, 16), 3.0))   # all "Nearly every day"
    cls_notatall, proba_notatall = _pred(np.full((1, 16), 0.0))   # all "Not at all"
    cls_half,    proba_half      = _pred(np.full((1, 16), 2.0))   # all "Around half"
    cls_several, proba_several   = _pred(np.full((1, 16), 1.0))   # all "Several days"

    return JSONResponse({
        "model_type": type(model).__name__,
        "model_classes_raw": [str(c) for c in model.classes_],
        "model_classes_decoded": [CLASS_DECODE.get(str(c), str(c)) for c in model.classes_],
        "n_features": len(FEATURES),
        "label_map": LABEL_MAP,
        "sanity_checks": {
            "all_Nearly_every_day_3": {
                "prediction": cls_nearly, "proba": proba_nearly,
                "expected": "High"
            },
            "all_Not_at_all_0": {
                "prediction": cls_notatall, "proba": proba_notatall,
                "expected": "Low"
            },
            "all_Around_half_2": {
                "prediction": cls_half, "proba": proba_half,
                "expected": "High or Medium"
            },
            "all_Several_days_1": {
                "prediction": cls_several, "proba": proba_several,
                "expected": "Low or Medium"
            },
        },
        "note": "With correct ordinal encoding: Nearly every day=3 → High, Not at all=0 → Low"
    })


@app.post("/predict")
async def predict(req: PredictionRequest):
    import logging
    log = logging.getLogger("ppd.predict")

    # ── build ordered feature vector ──────────────────────────────────────────
    vector = []
    missing = []
    for feat in FEATURES:
        txt = req.answers.get(feat)
        if txt is None:
            missing.append(repr(feat))
            txt = "Not at all"
        vector.append(LABEL_MAP.get(txt, 0))

    log.warning("=== /predict called ===")
    log.warning("Answers received : %d keys", len(req.answers))
    log.warning("Features expected: %d", len(FEATURES))
    if missing:
        log.warning("MISSING features (defaulted to 'Not at all'): %s", missing)
    log.warning("Encoded vector  : %s", vector)

    X     = np.array([vector], dtype=float)
    X_sc  = scaler.transform(X)
    log.warning("X shape: %s  |  X_scaled[0]: %s", X.shape, X_sc[0].tolist())

    # ── predict ───────────────────────────────────────────────────────────────
    raw_pred   = str(model.predict(X_sc)[0])
    pred_class = CLASS_DECODE.get(raw_pred, raw_pred)
    proba      = model.predict_proba(X_sc)[0]
    raw_classes = [str(c) for c in model.classes_]
    classes    = [CLASS_DECODE.get(c, c) for c in raw_classes]
    proba_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    log.warning("Raw prediction: %s → Decoded: %s", raw_pred, pred_class)
    log.warning("Probabilities: %s", proba_dict)

    # ── SHAP values via LinearExplainer ───────────────────────────────────────
    log.warning("Running SHAP LinearExplainer...")
    shap_exp = explainer(X_sc)
    shap_3d  = shap_exp.values
    log.warning("shap_exp.values type : %s", type(shap_3d))
    log.warning("shap_exp.values shape: %s", getattr(shap_3d, 'shape', 'NO SHAPE'))

    # Normalise to (1, 16, 3) — LinearExplainer may return (1, 16, 3) directly
    # or a list of (1,16) arrays (one per class in older SHAP versions)
    if isinstance(shap_3d, list):
        log.warning("SHAP returned list of %d arrays — stacking to 3D", len(shap_3d))
        shap_3d = np.stack([a[0] if a.ndim == 2 else a for a in shap_3d], axis=-1)
        shap_3d = shap_3d[np.newaxis, ...]   # (1, 16, n_classes)
    else:
        shap_3d = np.array(shap_3d)
        if shap_3d.ndim == 2:
            # (n_samples, n_features) — binary case, unlikely but safe
            shap_3d = shap_3d[:, :, np.newaxis]

    log.warning("Final shap_3d shape: %s", shap_3d.shape)

    pred_idx = classes.index(pred_class)
    sv = shap_3d[0, :, pred_idx]
    log.warning("sv (predicted class SHAP): %s", sv)

    # per-class SHAP dict
    all_sv = {classes[i]: shap_3d[0, :, i].tolist() for i in range(shap_3d.shape[2])}

    # ── top-3 by |SHAP| for predicted class ──────────────────────────────────
    top3_idx = np.argsort(np.abs(sv))[-3:][::-1]
    log.warning("Top-3 feature indices: %s  names: %s",
                top3_idx.tolist(), [FEATURES[i] for i in top3_idx])
    top3 = []
    for rank_pos, idx in enumerate(top3_idx):
        feat    = FEATURES[idx]
        info    = FEATURE_INFO.get(feat, {})
        enc_val = vector[idx]
        txt_val = next((k for k, v in LABEL_MAP.items() if v == enc_val), "Not at all")
        shap_val = float(sv[idx])
        log.warning("  Rank %d | feat=%s | shap=%.5f | answer=%s",
                    rank_pos+1, feat[:40], shap_val, txt_val)
        top3.append({
            "rank":         rank_pos + 1,
            "feature":      feat,
            "short":        info.get("short", feat),
            "scale":        info.get("scale", ""),
            "meaning":      info.get("meaning", ""),
            "shap_value":   round(shap_val, 5),
            "answer":       txt_val,
            "shap_all_cls": {c: round(float(shap_3d[0, idx, i]), 5)
                             for i, c in enumerate(classes)},
        })

    # ── global importance (mean |SHAP| across all classes) ────────────────────
    global_imp = {}
    for i, feat in enumerate(FEATURES):
        global_imp[feat] = round(float(np.mean(np.abs(shap_3d[0, i, :]))), 5)

    log.warning("=== Response ready — returning JSON ===")
    return JSONResponse({
        "prediction":        pred_class,
        "probabilities":     proba_dict,
        "top3_features":     top3,
        "global_importance": global_imp,
        "all_shap":          all_sv,
        "classes":           classes,
        "feature_order":     FEATURES,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
