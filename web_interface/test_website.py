"""
PPD Web Interface — Deep Debug & Test Suite
===========================================
HOW TO USE:
  1. Open ONE terminal  →  cd web_interface  →  python app.py
  2. Open ANOTHER terminal  →  python test_website.py

The script tests every layer:
  Layer 1: Model files exist and load correctly
  Layer 2: Scaler transforms sanity
  Layer 3: Model predictions (raw, encoded correctly)
  Layer 4: SHAP LinearExplainer values
  Layer 5: Live HTTP — /features, /debug, /predict
  Layer 6: Edge cases (missing keys, wrong keys, partial answers)
  Layer 7: Clinical sanity (Nearly every day → High, Not at all → Low)
"""

import json
import sys
import time
import os
import requests
import numpy as np

BASE_URL = "http://localhost:8080"
PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠  WARN\033[0m"
INFO = "\033[94mℹ  INFO\033[0m"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    results.append((label, condition))
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────
# LAYER 1: File existence
# ──────────────────────────────────────────────────────────────
section("LAYER 1 — Model files")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PKG_DIR   = os.path.join(BASE_DIR, "..", "shap_lr_package")
MODEL_PATH  = os.path.join(PKG_DIR, "lr_model.pkl")
SCALER_PATH = os.path.join(PKG_DIR, "lr_scaler.pkl")
TMPL_PATH   = os.path.join(BASE_DIR, "templates", "index.html")

check("lr_model.pkl exists",   os.path.isfile(MODEL_PATH),   MODEL_PATH)
check("lr_scaler.pkl exists",  os.path.isfile(SCALER_PATH),  SCALER_PATH)
check("index.html exists",     os.path.isfile(TMPL_PATH),    TMPL_PATH)


# ──────────────────────────────────────────────────────────────
# LAYER 2: Load model + scaler
# ──────────────────────────────────────────────────────────────
section("LAYER 2 — Load model & scaler")
try:
    import joblib, shap
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    check("Model loaded", True, f"Type: {type(model).__name__}")
    check("Scaler loaded", True, f"Type: {type(scaler).__name__}")
    check("Model has 3 classes", len(model.classes_) == 3,
          f"classes_={model.classes_}")
    check("Model n_features == 16", model.n_features_in_ == 16,
          f"n_features_in_={model.n_features_in_}")
    check("Scaler n_features == 16", scaler.n_features_in_ == 16,
          f"n_features_in_={scaler.n_features_in_}")
except Exception as e:
    check("Model/scaler load", False, str(e))
    print("\n[FATAL] Cannot proceed without model — fix the load error first.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# LAYER 3: Scaler sanity
# ──────────────────────────────────────────────────────────────
section("LAYER 3 — Scaler transform sanity")

zero_raw   = np.zeros((1, 16))
zero_sc    = scaler.transform(zero_raw)
three_raw  = np.full((1, 16), 3.0)
three_sc   = scaler.transform(three_raw)

check("Zero vector transforms without error",
      zero_sc is not None, f"scaled[0][:4]={zero_sc[0][:4].tolist()}")
check("Scaled zero != raw zero (scaler is non-trivial)",
      not np.allclose(zero_sc, zero_raw),
      f"mean_scale_val={zero_sc[0].mean():.4f}")
check("Scaled 3-vector != raw 3-vector",
      not np.allclose(three_sc, three_raw),
      f"mean_scale_val={three_sc[0].mean():.4f}")
check("Scaler does not produce NaN or Inf",
      np.all(np.isfinite(zero_sc)) and np.all(np.isfinite(three_sc)),
      "NaN/Inf check passed")


# ──────────────────────────────────────────────────────────────
# LAYER 4: Model prediction sanity (no server needed)
# ──────────────────────────────────────────────────────────────
section("LAYER 4 — Direct model prediction sanity")

CLASS_DECODE = {"0": "High", "1": "Low", "2": "Medium"}

def predict_direct(raw_val):
    X    = np.full((1, 16), raw_val)
    X_sc = scaler.transform(X)
    raw  = str(model.predict(X_sc)[0])
    cls  = CLASS_DECODE.get(raw, raw)
    prob = {CLASS_DECODE.get(str(c), str(c)): round(float(p), 4)
            for c, p in zip(model.classes_, model.predict_proba(X_sc)[0])}
    return cls, prob

cls_3, prob_3 = predict_direct(3.0)
cls_0, prob_0 = predict_direct(0.0)
cls_2, prob_2 = predict_direct(2.0)
cls_1, prob_1 = predict_direct(1.0)

print(f"\n  all Nearly every day (3): → {cls_3}  proba={prob_3}")
print(f"  all Not at all       (0): → {cls_0}  proba={prob_0}")
print(f"  all Around half      (2): → {cls_2}  proba={prob_2}")
print(f"  all Several days     (1): → {cls_1}  proba={prob_1}")

check("Nearly every day (3) → High",  cls_3 == "High",
      f"Got: {cls_3} | proba: {prob_3}")
check("Not at all (0) → Low",         cls_0 == "Low",
      f"Got: {cls_0} | proba: {prob_0}")
check("Around half (2) → High or Medium",
      cls_2 in ("High", "Medium"),
      f"Got: {cls_2} | proba: {prob_2}")
check("Several days (1) → Low or Medium",
      cls_1 in ("Low", "Medium"),
      f"Got: {cls_1} | proba: {prob_1}")

# Class decode consistency
for raw_c in model.classes_:
    decoded = CLASS_DECODE.get(str(raw_c))
    check(f"CLASS_DECODE[{raw_c}] is defined",
          decoded is not None, f"decoded={decoded}")


# ──────────────────────────────────────────────────────────────
# LAYER 5: SHAP LinearExplainer sanity
# ──────────────────────────────────────────────────────────────
section("LAYER 5 — SHAP LinearExplainer sanity")

try:
    _bg       = scaler.transform(np.zeros((1, 16)))
    explainer = shap.LinearExplainer(model, shap.maskers.Independent(_bg))
    check("LinearExplainer created", True)

    X_test   = scaler.transform(np.full((1, 16), 3.0))
    shap_exp = explainer(X_test)
    sv       = shap_exp.values

    check("SHAP values returned", sv is not None)

    # Normalise to 3D
    if isinstance(sv, list):
        sv = np.stack([a[0] if a.ndim==2 else a for a in sv], axis=-1)[np.newaxis]
    else:
        sv = np.array(sv)
        if sv.ndim == 2:
            sv = sv[:, :, np.newaxis]

    check("SHAP shape is (1, 16, 3)",
          sv.shape == (1, 16, 3), f"shape={sv.shape}")
    check("SHAP values are finite",
          np.all(np.isfinite(sv)), "No NaN/Inf in SHAP output")

    # For High-risk input, High class SHAP should dominate positively
    high_idx = [i for i, c in enumerate(CLASS_DECODE.values()) if c == "High"]
    if not high_idx:
        # find by raw class
        cls_list = [CLASS_DECODE.get(str(c), str(c)) for c in model.classes_]
        high_idx_in_3d = cls_list.index("High") if "High" in cls_list else None
    else:
        cls_list = [CLASS_DECODE.get(str(c), str(c)) for c in model.classes_]
        high_idx_in_3d = cls_list.index("High") if "High" in cls_list else None

    if high_idx_in_3d is not None:
        sv_high = sv[0, :, high_idx_in_3d]
        top_feat_idx = int(np.argmax(np.abs(sv_high)))
        print(f"\n  Top SHAP feature for High-risk input: feature index {top_feat_idx}")
        print(f"  Mean |SHAP| per class: {np.mean(np.abs(sv[0]), axis=0).tolist()}")
        check("Top SHAP feature has positive value for High class (Nearly every day input)",
              sv_high[top_feat_idx] > 0,
              f"shap_val={sv_high[top_feat_idx]:.4f}")

except Exception as e:
    check("SHAP explainer", False, str(e))


# ──────────────────────────────────────────────────────────────
# LAYER 6: Live HTTP tests (server must be running)
# ──────────────────────────────────────────────────────────────
section("LAYER 6 — Live HTTP (server must be running on :8080)")

def get(path, timeout=5):
    try:
        r = requests.get(BASE_URL + path, timeout=timeout)
        return r.status_code, r
    except requests.exceptions.ConnectionError:
        return None, None

def post(path, body, timeout=10):
    try:
        r = requests.post(BASE_URL + path, json=body, timeout=timeout)
        return r.status_code, r
    except requests.exceptions.ConnectionError:
        return None, None

# Check server is up
status, r = get("/")
if status is None:
    print(f"\n  {WARN} Server not reachable at {BASE_URL}")
    print("  → Start the server first:  cd web_interface && python app.py")
    print("  (Skipping all HTTP tests)\n")
else:
    # GET /
    check("GET / returns 200",     status == 200, f"status={status}")
    check("GET / returns HTML",    "text/html" in r.headers.get("content-type",""),
          f"content-type={r.headers.get('content-type')}")

    # GET /features
    status2, r2 = get("/features")
    check("GET /features returns 200", status2 == 200, f"status={status2}")
    if r2:
        data = r2.json()
        check("features list has 16 items", len(data.get("features",[])) == 16,
              f"count={len(data.get('features',[]))}")
        check("options list has 4 items",   len(data.get("options",[])) == 4,
              f"options={data.get('options')}")
        check("info dict has 16 entries",   len(data.get("info",{})) == 16,
              f"count={len(data.get('info',{}))}")

    # GET /debug
    status3, r3 = get("/debug")
    check("GET /debug returns 200", status3 == 200, f"status={status3}")
    if r3:
        dbg = r3.json()
        print(f"\n  /debug sanity checks:")
        for k, v in dbg.get("sanity_checks", {}).items():
            expected = v.get("expected","")
            prediction = v.get("prediction","")
            ok = prediction in expected
            sym = "✅" if ok else "❌"
            print(f"    {sym} {k}: predicted={prediction} (expected: {expected})")
            results.append((f"debug/{k}", ok))

    # POST /predict — High-risk case (all Nearly every day)
    features_list = data.get("features", []) if 'data' in dir() and data else []
    if not features_list:
        status_f, r_f = get("/features")
        features_list = r_f.json().get("features", []) if r_f else []

    if features_list:
        high_answers = {f: "Nearly every day" for f in features_list}
        s, r_pred = post("/predict", {"answers": high_answers})
        check("POST /predict (all High) returns 200", s == 200, f"status={s}")
        if r_pred and s == 200:
            p = r_pred.json()
            check("High-risk input → prediction == High",
                  p.get("prediction") == "High",
                  f"prediction={p.get('prediction')}  proba={p.get('probabilities')}")
            check("top3_features has 3 items",
                  len(p.get("top3_features",[])) == 3,
                  f"count={len(p.get('top3_features',[]))}")
            check("global_importance has 16 items",
                  len(p.get("global_importance",{})) == 16,
                  f"count={len(p.get('global_importance',{}))}")
            top3 = p.get("top3_features", [])
            if top3:
                print(f"\n  Top-3 features for all-High input:")
                for t in top3:
                    print(f"    Rank {t['rank']}: {t['short']} | SHAP={t['shap_value']} | answer={t['answer']}")

        # POST /predict — Low-risk case (all Not at all)
        low_answers = {f: "Not at all" for f in features_list}
        s2, r_low = post("/predict", {"answers": low_answers})
        check("POST /predict (all Low) returns 200", s2 == 200, f"status={s2}")
        if r_low and s2 == 200:
            p2 = r_low.json()
            check("Low-risk input → prediction == Low",
                  p2.get("prediction") == "Low",
                  f"prediction={p2.get('prediction')}  proba={p2.get('probabilities')}")

        # POST /predict — realistic mixed case
        mixed = {f: "Not at all" for f in features_list}
        # Set a few severe answers
        severe_items = [
            "Trouble falling or staying asleep, or sleeping too much",
            "You have been so unhappy that you have been crying",
            "You have felt scared or panicky for no good reason",
            "Feeling down, depressed, or hopeless",
        ]
        for item in severe_items:
            if item in mixed:
                mixed[item] = "Nearly every day"
        s3, r_mix = post("/predict", {"answers": mixed})
        check("POST /predict (mixed) returns 200", s3 == 200)
        if r_mix and s3 == 200:
            p3 = r_mix.json()
            print(f"\n  Mixed case (4 severe, rest none): → {p3.get('prediction')}")
            print(f"  Probabilities: {p3.get('probabilities')}")

        # POST /predict — empty answers (should default all to Not at all)
        s4, r_empty = post("/predict", {"answers": {}})
        check("POST /predict (empty answers) returns 200", s4 == 200,
              "Server must not crash on empty input")
        if r_empty and s4 == 200:
            p4 = r_empty.json()
            check("Empty answers → defaults to Low",
                  p4.get("prediction") == "Low",
                  f"got: {p4.get('prediction')}")

        # POST /predict — partial answers (only 3 keys)
        partial = {features_list[0]: "Nearly every day",
                   features_list[1]: "Nearly every day",
                   features_list[2]: "Nearly every day"}
        s5, r_part = post("/predict", {"answers": partial})
        check("POST /predict (partial 3/16 answers) returns 200", s5 == 200)

        # POST /predict — wrong key (typo)
        bad = {"NOT A REAL FEATURE": "Nearly every day"}
        s6, r_bad = post("/predict", {"answers": bad})
        check("POST /predict (unknown key) returns 200 without crashing", s6 == 200)


# ──────────────────────────────────────────────────────────────
# LAYER 7: SHAP value direction check (clinical logic)
# ──────────────────────────────────────────────────────────────
section("LAYER 7 — SHAP direction: severe → higher SHAP for High class")

try:
    _bg = scaler.transform(np.zeros((1, 16)))
    explainer_check = shap.LinearExplainer(model, shap.maskers.Independent(_bg))
    cls_list = [CLASS_DECODE.get(str(c), str(c)) for c in model.classes_]

    high_idx = cls_list.index("High") if "High" in cls_list else None
    low_idx  = cls_list.index("Low")  if "Low"  in cls_list else None

    # Training column indices (from chi2_optimal_features.txt training order):
    #   idx 7  = crying       coef_High = +2.137  → SHAP positive for High ✓
    #   idx 15 = sleep        coef_High = +2.318  → SHAP positive for High ✓
    # (idx 8=laugh, idx 9=look_forward are reversed items with negative High coef)
    for feat_idx in [7, 15]:   # crying (training idx 7), sleep (training idx 15)
        arr_severe = np.zeros((1, 16))
        arr_none   = np.zeros((1, 16))
        arr_severe[0, feat_idx] = 3.0

        sv_s = explainer_check(scaler.transform(arr_severe)).values
        sv_n = explainer_check(scaler.transform(arr_none)).values

        if isinstance(sv_s, list):
            sv_s = np.stack([a[0] if a.ndim==2 else a for a in sv_s], axis=-1)[np.newaxis]
            sv_n = np.stack([a[0] if a.ndim==2 else a for a in sv_n], axis=-1)[np.newaxis]
        else:
            sv_s = np.array(sv_s)
            sv_n = np.array(sv_n)
            if sv_s.ndim == 2: sv_s = sv_s[:, :, np.newaxis]
            if sv_n.ndim == 2: sv_n = sv_n[:, :, np.newaxis]

        if high_idx is not None:
            shap_high_s = sv_s[0, feat_idx, high_idx]
            shap_high_n = sv_n[0, feat_idx, high_idx]
            feat_name = ["crying", "sleep disturbance"][feat_idx == 15]
            check(
                f"SHAP[{feat_name}, High class]: severe(3) > none(0)",
                shap_high_s > shap_high_n,
                f"severe={shap_high_s:.4f} vs none={shap_high_n:.4f}"
            )

        if low_idx is not None:
            shap_low_s = sv_s[0, feat_idx, low_idx]
            shap_low_n = sv_n[0, feat_idx, low_idx]
            feat_name = ["crying", "sleep disturbance"][feat_idx == 15]
            check(
                f"SHAP[{feat_name}, Low class]: severe(3) < none(0)",
                shap_low_s < shap_low_n,
                f"severe={shap_low_s:.4f} vs none={shap_low_n:.4f}"
            )

except Exception as e:
    check("SHAP direction check", False, str(e))


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
section("SUMMARY")
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n  Total checks : {total}")
print(f"  {PASS} Passed : {passed}")
if failed:
    print(f"  {FAIL} Failed : {failed}")
    print("\n  Failed checks:")
    for label, ok in results:
        if not ok:
            print(f"    ❌  {label}")
else:
    print(f"\n  🎉  All checks passed! Your website is working correctly.")

print()
