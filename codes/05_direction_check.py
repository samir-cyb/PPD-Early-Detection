"""
05_direction_check.py
======================
Why this script exists
-----------------------
04_feature_engineering.py's WEIGHTED Maternal Mental-Health Risk Index
(MHRI) became strongly significant (great), but its direction looks
BACKWARDS: the High-EPDS group has a LOWER weighted-MHRI mean (3.04) than
the Low-EPDS group (5.14). Since 'Abuse' carries 63% of the weight in that
index, and we assumed Abuse={"No": 0, "Yes": 1} (Yes = higher risk score),
a flipped result this strong suggests our assumed direction for the
'Abuse' answer may not match what "Yes" actually means in this survey (it
could be a translation/labeling quirk, e.g. "Yes" meaning "no abuse
experienced" rather than "abuse experienced").

Rather than guess, this script empirically checks the assumed direction of
EVERY ordinal mapping used in the 4 composite indices, by printing the
actual PPD_binary positive-rate for each category, ordered by the score we
assigned it. If our assumption is right, the PPD-positive rate should rise
(or fall) monotonically alongside the assigned score. If it doesn't, that
column's mapping needs to be flipped in 04_feature_engineering.py.

Usage
-----
    python 05_direction_check.py
"""

import os
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

df = pd.read_csv(CLEANED_PATH)
print(f"Loaded: {df.shape}\n")

# column -> (assumed ordinal map [POST-FIX, matching 04_feature_engineering.py],
#            expected_direction: 'increasing' for RISK-index sub-features
#            (MHRI, NSI -- higher score should mean MORE PPD), 'decreasing'
#            for PROTECTIVE-index sub-features (SSI, ESI -- higher score
#            should mean LESS PPD, since higher = more support/stability).
CHECKS = {
    # --- MHRI sub-features (risk index -> expect increasing) ---
    "Abuse": ({"Yes": 0, "No": 1}, "increasing"),
    "Depression before pregnancy (PHQ2)": ({"Negative": 0, "Positive": 1},
                                            "increasing"),
    "Depression during pregnancy (PHQ2)": ({"Negative": 0, "Positive": 1},
                                            "increasing"),
    "Disease before pregnancy": ({"No Disease": 0, "Non-Chronic Disease": 1,
                                   "Chronic Disease": 2}, "increasing"),
    "History of pregnancy loss": ({"No Pregnancy Loss": 0, "Miscarriage": 1,
                                    "Still-born Delivery": 2}, "increasing"),
    # --- NSI sub-features (risk index -> expect increasing) ---
    "Mode of delivery": ({"Normal Delivery": 0, "Caesarean Section": 1},
                          "increasing"),
    "Birth complications": ({"No": 0, "Yes": 1}, "increasing"),
    "Newborn illness": ({"No": 0, "Yes": 1}, "increasing"),
    "Worry about newborn": ({"No": 0, "Yes": 1}, "increasing"),
    # --- SSI sub-features (protective index -> expect decreasing) ---
    "Family type": ({"Joint": 0, "Nuclear": 1}, "decreasing"),
    "Relationship with the in-laws": ({"Bad": 0, "Poor": 1, "Neutral": 2,
                                        "Good": 3, "Friendly": 4},
                                       "decreasing"),
    "Relationship with husband": ({"Bad": 0, "Poor": 1, "Neutral": 2,
                                    "Good": 3, "Friendly": 4}, "decreasing"),
    "Number of household members": ({"9 or more": 0, "6 to 8": 1,
                                      "2 to 5": 2}, "decreasing"),
    "Received Support": ({"Low": 0, "Medium": 1, "High": 2}, "decreasing"),
    # --- ESI sub-features (protective index -> expect decreasing) ---
    "Current monthly income": ({"No Personal Income": 0, "Less than 5000": 1,
                                 "5000 to 10000": 2, "10000 to 20000": 3,
                                 "20000 to 30000": 4, "More than 30000": 5},
                                "decreasing"),
    "Education Level": ({"University": 1, "College": 2, "High School": 3,
                          "Primary School": 4}, "decreasing"),
}

print("For each column: PPD_binary positive-rate (% with PPD=1) per "
      "category, ORDERED by the ordinal score we assigned it (POST-FIX "
      "maps, matching the corrected 04_feature_engineering.py).\n"
      "'increasing' columns belong to a RISK index (MHRI/NSI) -- higher "
      "score should mean MORE PPD.\n"
      "'decreasing' columns belong to a PROTECTIVE index (SSI/ESI) -- "
      "higher score should mean LESS PPD.\n")

for col, (mapping, expected) in CHECKS.items():
    if col not in df.columns:
        print(f"[skip] column not found: {col}")
        continue
    rate = df.groupby(col)["PPD_binary"].agg(["mean", "count"])
    rate["assigned_score"] = rate.index.map(mapping)
    rate = rate.sort_values("assigned_score")
    rate["mean"] = (rate["mean"] * 100).round(1)
    rate = rate.rename(columns={"mean": "PPD_positive_rate_%"})
    print(f"--- {col} (expected: {expected}) ---")
    print(rate[["assigned_score", "PPD_positive_rate_%", "count"]]
          .to_string())
    vals = rate["PPD_positive_rate_%"].values
    increasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    decreasing = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    if expected == "increasing":
        ok = increasing
    else:
        ok = decreasing
    if ok:
        print(f"  -> Matches expected '{expected}' trend: direction is "
              f"CORRECT.")
    elif increasing or decreasing:
        print(f"  -> Trend is monotonic but OPPOSITE of expected "
              f"'{expected}' -- direction is still WRONG, needs another "
              f"look.")
    else:
        print("  -> NOT monotonic (mixed trend, often just small-sample "
              "noise in one category) -- not necessarily wrong.")
    print()
