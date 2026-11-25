"""Vérifie que l'application donne les mêmes prix que le notebook pour les 4 maisons héritées.

Usage :  python backend/check_predictions.py     (après python backend/train.py)
"""
from pathlib import Path
import pandas as pd, joblib
from ml_pipeline import encoder

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "model_pipeline.joblib")
inh = pd.read_csv(ROOT / "data" / "inherited_houses.csv")
prix = model.predict(encoder(inh))
for i, p in enumerate(prix, 1):
    print(f"Maison {i} : {p:,.0f} $")
print(f"Total : {prix.sum():,.0f} $")
