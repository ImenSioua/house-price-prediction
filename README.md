# Prédiction du prix des maisons — Projet Machine Learning (ITBS)

Application web qui estime le prix de vente d'une maison à partir de ses caractéristiques.
Modèle : **Gradient Boosting** (scikit-learn), choisi parmi 8 modèles de régression par validation croisée.

| Mesure (292 maisons jamais vues) | Valeur   |
| -------------------------------- | -------- |
| R² test                          | 0,905    |
| RMSE                             | 26 967 $ |
| MAE                              | 17 181 $ |

Données : Kaggle « House Prices: Advanced Regression Techniques » (1460 maisons) + 4 maisons héritées sans prix.

## Contenu du dépôt

- `ML.ipynb` : notebook complet (exploration, prétraitement, Feature Engineering, 8 modèles, évaluation, 4 maisons héritées).
- `backend/ml_pipeline.py` : même prétraitement et même modèle que le notebook.
- `backend/train.py` : entraîne le modèle et le sauvegarde dans `models/`.
- `backend/app.py` : API FastAPI (`/predict`, `/health`) qui sert aussi l'interface web.
- `backend/check_predictions.py` : vérifie les prix des 4 maisons héritées.
- `frontend/index.html` : formulaire de saisie.
- `data/` : les deux fichiers CSV.

## Lancer en local

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac : source venv/bin/activate)
pip install -r backend/requirements.txt
python backend/train.py          # crée models/model_pipeline.joblib (quelques secondes)
python backend/check_predictions.py   # Maison 1 : 134 895 $ ... Total : 659 401 $
cd backend
uvicorn app:app --reload
```

Puis ouvrir http://127.0.0.1:8000

## Déploiement sur Render

- Type : Web Service, branche `main`
- Build Command : `pip install -r backend/requirements.txt && python backend/train.py`
- Start Command : `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

## Remarque

L'intervalle affiché est « prix ± RMSE » : c'est une erreur typique mesurée sur le jeu de test, pas un intervalle de confiance statistique.
