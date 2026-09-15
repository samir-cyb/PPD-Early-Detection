# PPD Risk Assessment — Web Interface

AI-powered postpartum depression screening using Gradient Boosting + SHAP explainability.

## Quick Start

```
Double-click  run.bat
Then open     http://localhost:8000
```

## Requirements

- Python 3.9+
- The `shap_package/` folder (with `gb_model.pkl`) must be one level up from `web_interface/`

## File Structure

```
dataset with exit paper/
├── shap_package/
│   ├── gb_model.pkl          ← trained GradientBoostingClassifier
│   └── gb_scaler.pkl
└── web_interface/
    ├── app.py                ← FastAPI backend
    ├── requirements.txt
    ├── run.bat               ← Windows start script
    └── templates/
        └── index.html        ← Single-page UI
```

## API Endpoints

| Endpoint          | Method | Description                          |
|-------------------|--------|--------------------------------------|
| `/`               | GET    | Serve the web UI                     |
| `/features`       | GET    | Return feature list + info           |
| `/predict`        | POST   | Predict + return SHAP explanation    |

### POST /predict — Request

```json
{
  "answers": {
    "You have been so unhappy that you have been crying": "Nearly every day",
    "You have felt scared or panicky for no good reason": "Several days",
    ...
  }
}
```

### POST /predict — Response

```json
{
  "prediction": "High",
  "probabilities": { "High": 0.82, "Medium": 0.11, "Low": 0.07 },
  "top3_features": [
    {
      "rank": 1,
      "feature": "You have been so unhappy that you have been crying",
      "short": "Crying / tearfulness",
      "scale": "EPDS",
      "meaning": "...",
      "shap_value": 0.071,
      "answer": "Nearly every day",
      "shap_all_cls": { "High": 0.071, "Medium": -0.02, "Low": -0.05 }
    },
    ...
  ],
  "global_importance": { "feature_name": 0.0585, ... },
  "classes": ["High", "Low", "Medium"]
}
```

## Model Details

- **Model**: GradientBoostingClassifier (sklearn)
- **Target**: EPDS Result — 3 classes (Low=0–9, Medium=10–12, High=13–30)
- **Features**: 17 selected EPDS + PHQ-9 questionnaire items
- **CV Accuracy**: 0.926
- **SHAP**: TreeExplainer (real-time, per-sample)
- **Encoding**: LabelEncoder alphabetical — "Around half the days"=0, "Nearly every day"=1, "Not at all"=2, "Several days"=3
