"""Entraîne le modèle final (même méthode que le notebook) et le sauvegarde dans ../models/.

Usage :  python backend/train.py        (depuis le dossier du projet)
"""
import json, platform
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from ml_pipeline import FEATURES, DERIVED, ORDINAL, REF_YEAR, SEED, TARGET, build_pipeline, encoder

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "house_prices_records.csv")
X, y = encoder(df), df[TARGET]

# 1) Évaluation honnête : 80 % / 20 %, le test n'est jamais vu pendant l'entraînement
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True)
pred = build_pipeline().fit(X_tr, y_tr).predict(X_te)
metrics = {"r2": round(float(r2_score(y_te, pred)), 4),
           "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
           "mae": round(float(mean_absolute_error(y_te, pred)), 2)}

# 2) Modèle final : réentraîné sur les 1460 maisons
final = build_pipeline().fit(X, y)
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(final, ROOT / "models" / "model_pipeline.joblib")
meta = {"modele": "Gradient Boosting", "features": FEATURES, "derived_features": DERIVED,
        "ordinal_mapping": ORDINAL, "ref_year": REF_YEAR, "seed": SEED,
        "metriques_test": metrics, "n_train": len(X_tr), "n_test": len(X_te), "n_total": len(df),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__}}
(ROOT / "models" / "model_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("Modèle sauvegardé. Performance sur 292 maisons de test :", metrics)
