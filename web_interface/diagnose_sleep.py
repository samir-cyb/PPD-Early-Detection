"""
Diagnose: why does sleep disturbance have a negative SHAP for the High class?
Run this standalone — no server needed.
"""
import os, sys
import numpy as np
import joblib, shap

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PKG_DIR     = os.path.join(BASE_DIR, "..", "shap_lr_package")
model       = joblib.load(os.path.join(PKG_DIR, "lr_model.pkl"))
scaler      = joblib.load(os.path.join(PKG_DIR, "lr_scaler.pkl"))

CLASS_DECODE = {"0": "High", "1": "Low", "2": "Medium"}
FEATURES = [
    "You have been so unhappy that you have been crying",
    "You have felt scared or panicky for no good reason",
    "You have felt sad or miserable",
    "Little interest or pleasure in doing things",
    "You have been anxious or worried for no good reason",
    "Feeling down, depressed, or hopeless",
    "Trouble concentrating on things",
    "Feeling bad about yourself...",
    "Things have been getting to you",
    "Trouble falling or staying asleep, or sleeping too much",   # index 9
    "You have blamed myself unnecessarily...",
    "You have looked forward with enjoyment to things",
    "You have been able to laugh and see the funny side of things",
    "Feeling tired or having little energy",
    "Thoughts that you would be better off dead...",
    "The thought of harming yourself has occurred",
]

cls_names = [CLASS_DECODE.get(str(c), str(c)) for c in model.classes_]
print("Classes:", cls_names)   # ['High', 'Low', 'Medium']

# ── 1. Print LR coefficients for sleep (index 9) ──────────────────────────────
print("\n=== LR coef_ for sleep disturbance (feature idx 9) ===")
for i, cls in enumerate(cls_names):
    coef = model.coef_[i, 9]
    direction = "→ RAISES" if coef > 0 else "→ LOWERS"
    print(f"  {cls:8s}: coef={coef:+.4f}  {direction} P({cls})")

# ── 2. Print ALL coef_ signs for sleep vs crying ──────────────────────────────
print("\n=== Comparison: coef signs for all 16 features ===")
print(f"{'Feature':<50} {'High':>8} {'Low':>8} {'Medium':>8}")
print("-"*80)
for i, feat in enumerate(FEATURES):
    h = model.coef_[0, i]
    l = model.coef_[1, i]
    m = model.coef_[2, i]
    flag = "  ← REVERSED!" if h < 0 and feat == FEATURES[9] else ""
    print(f"  {feat[:48]:<50} {h:+.4f}  {l:+.4f}  {m:+.4f}{flag}")

# ── 3. What does the model predict for realistic cases? ───────────────────────
def predict(vec, label=""):
    X    = np.array([vec], dtype=float)
    X_sc = scaler.transform(X)
    raw  = str(model.predict(X_sc)[0])
    cls  = CLASS_DECODE.get(raw, raw)
    prob = {CLASS_DECODE.get(str(c), str(c)): round(float(p), 4)
            for c, p in zip(model.classes_, model.predict_proba(X_sc)[0])}
    print(f"  {label:55s} → {cls}  {prob}")
    return cls

print("\n=== Realistic prediction cases ===")
base = [0]*16

# Only sleep severe
v = base.copy(); v[9] = 3
predict(v, "Sleep only = Nearly every day")

# Only crying severe
v = base.copy(); v[0] = 3
predict(v, "Crying only = Nearly every day")

# Crying + panic + hopelessness severe (no sleep)
v = base.copy(); v[0]=3; v[1]=3; v[5]=3
predict(v, "Crying + Panic + Hopelessness = 3 (no sleep)")

# Crying + panic + hopelessness + sleep severe
v = base.copy(); v[0]=3; v[1]=3; v[5]=3; v[9]=3
predict(v, "Crying + Panic + Hopelessness + Sleep = 3")

# Top 3 from SHAP global (sleep #1, crying #2, panic #3) all severe
v = base.copy(); v[0]=3; v[1]=3; v[9]=3
predict(v, "Sleep + Crying + Panic = 3")

# Same as above but sleep = 0
v = base.copy(); v[0]=3; v[1]=3; v[9]=0
predict(v, "Crying + Panic = 3 (sleep=0)")

# ── 4. SHAP for sleep alone ───────────────────────────────────────────────────
print("\n=== SHAP values when sleep=3, rest=0 ===")
_bg = scaler.transform(np.zeros((1, 16)))
expl = shap.LinearExplainer(model, shap.maskers.Independent(_bg))

arr = np.zeros((1, 16)); arr[0, 9] = 3
sv = expl(scaler.transform(arr)).values
if isinstance(sv, list):
    sv = np.stack([a[0] if a.ndim==2 else a for a in sv], axis=-1)[np.newaxis]
else:
    sv = np.array(sv)
    if sv.ndim == 2: sv = sv[:, :, np.newaxis]

print(f"  sleep SHAP → High  : {sv[0, 9, 0]:+.4f}  (negative = LOWERS High risk probability)")
print(f"  sleep SHAP → Low   : {sv[0, 9, 1]:+.4f}  (positive = RAISES Low risk probability)")
print(f"  sleep SHAP → Medium: {sv[0, 9, 2]:+.4f}")

print("\n=== Root cause summary ===")
h_coef = model.coef_[0, 9]
l_coef = model.coef_[1, 9]
if h_coef < 0 and l_coef > 0:
    print("""
  The Logistic Regression model learned a NEGATIVE coefficient for:
    sleep disturbance  →  High risk class
  and a POSITIVE coefficient for:
    sleep disturbance  →  Low risk class

  This means the LR model believes: more sleep disturbance = MORE likely Low, LESS likely High.

  Why? In this Bangladeshi dataset, ALL postpartum mothers (regardless of PPD severity)
  experience sleep disruption due to infant care. So 'Nearly every day' for sleep trouble
  is extremely common even in the Low-risk group, and the LR model learned this association
  from the data. After controlling for crying, panic, sadness (which strongly predict High),
  sleep alone becomes a marker of Low or Medium risk.

  Implication for the website:
  - When sleep=3 is submitted ALONE with other features near 0 → prediction pulls toward Low ✓
  - When sleep=3 is submitted TOGETHER with crying/panic/hopelessness=3 → sleep COUNTERACTS
    the other symptoms, potentially producing a LOW prediction instead of HIGH.

  This is a REAL model limitation documented in Chapter 4 (Limitation 4) and should be
  communicated to healthcare users.
""")
elif h_coef > 0:
    print("  Coefficient is POSITIVE — the SHAP failure was a test logic error, not a model error.")
