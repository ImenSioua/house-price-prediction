import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ml_pipeline import FEATURES, ORDINAL, encoder

app = FastAPI(title="House Price Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Chargement du modèle (créé par train.py à partir du notebook) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
FRONTEND = os.path.join(BASE_DIR, "..", "frontend", "index.html")

if not os.path.exists(os.path.join(MODEL_DIR, "model_pipeline.joblib")):
    raise RuntimeError("Modèle introuvable : lancez d'abord  python backend/train.py")
model = joblib.load(os.path.join(MODEL_DIR, "model_pipeline.joblib"))
with open(os.path.join(MODEL_DIR, "model_metadata.json"), encoding="utf-8") as f:
    meta = json.load(f)
RMSE = meta["metriques_test"]["rmse"]          # erreur typique mesurée sur 292 maisons jamais vues


@app.get("/")
def home():
    """Sert l'interface web (ou un message si le dossier frontend est absent)."""
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND)
    return {"message": "API House Price Prediction is running!"}


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["modele"], "r2_test": meta["metriques_test"]["r2"], "rmse_test": RMSE}


@app.post("/predict")
def predict(features: dict):
    try:
        # 1. Contrôles simples des entrées
        for col, mapping in ORDINAL.items():
            if features.get(col) not in (None, "") and features[col] not in mapping:
                return JSONResponse({"error": f"Valeur invalide pour {col} : {features[col]}. "
                                              f"Valeurs possibles : {list(mapping)}"}, status_code=422)
        row = pd.DataFrame([{k: (None if v == "" else v) for k, v in features.items() if k in FEATURES}])
        row = row.reindex(columns=FEATURES)
        missing = [c for c in ("OverallQual", "GrLivArea", "YearBuilt")
                   if pd.isna(pd.to_numeric(row[c], errors="coerce").iloc[0])]
        if missing:
            return JSONResponse({"error": "Champs obligatoires manquants : " + ", ".join(missing)}, status_code=422)
        if not 1 <= float(row["OverallQual"].iloc[0]) <= 10:
            return JSONResponse({"error": "OverallQual doit être compris entre 1 et 10."}, status_code=422)

        # 2. Même traitement que dans le notebook : encodage + Feature Engineering, puis le pipeline complet
        #    (imputation, standardisation, Gradient Boosting)
        price = float(model.predict(encoder(row))[0])

        return {"predicted_price": round(price, 2),
                "typical_error": RMSE,
                "range_low": round(price - RMSE, 2),
                "range_high": round(price + RMSE, 2)}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)
