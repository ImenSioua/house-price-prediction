"""Pipeline de prédiction : IDENTIQUE à celui du notebook (ML.ipynb).

Ce fichier est utilisé par train.py (entraînement) et par app.py (prédiction),
pour garantir que l'application applique exactement le même traitement que le notebook.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
REF_YEAR = 2024          # année de référence pour les âges (constante fixe)
TARGET = "SalePrice"
DROP = ["EnclosedPorch", "WoodDeckSF"]   # plus de 89 % de valeurs manquantes

# 21 variables d'origine, dans l'ordre du fichier CSV
FEATURES = ["1stFlrSF", "2ndFlrSF", "BedroomAbvGr", "BsmtExposure", "BsmtFinSF1", "BsmtFinType1",
            "BsmtUnfSF", "GarageArea", "GarageFinish", "GarageYrBlt", "GrLivArea", "KitchenQual",
            "LotArea", "LotFrontage", "MasVnrArea", "OpenPorchSF", "OverallCond", "OverallQual",
            "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]

# Variables catégorielles ordinales (l'ordre de la qualité est conservé)
ORDINAL = {
    "KitchenQual":  {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
}
DERIVED = ["TotalLivingArea", "PropertyAge", "RemodelAge",
           "BasementFinishRatio", "LivingToLotRatio", "AreaEfficiency"]


def encoder(frame: pd.DataFrame) -> pd.DataFrame:
    """Encodage ordinal + Feature Engineering (aucune statistique apprise, le prix n'est jamais utilisé)."""
    out = frame.reindex(columns=FEATURES).copy()
    for col in FEATURES:
        if col not in ORDINAL:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col, mapping in ORDINAL.items():
        out[col] = out[col].map(mapping)
    out["TotalLivingArea"] = out["1stFlrSF"] + out["2ndFlrSF"]
    out["PropertyAge"] = REF_YEAR - out["YearBuilt"]
    out["RemodelAge"] = REF_YEAR - out["YearRemodAdd"]
    out["BasementFinishRatio"] = (out["BsmtFinSF1"] / out["TotalBsmtSF"].where(out["TotalBsmtSF"] > 0)).fillna(0)
    out["LivingToLotRatio"] = out["GrLivArea"] / out["LotArea"]
    out["AreaEfficiency"] = out["GrLivArea"] / out["BedroomAbvGr"].clip(lower=1)
    return out


def build_pipeline() -> Pipeline:
    """Imputation (médiane / valeur la plus fréquente) -> standardisation -> Gradient Boosting."""
    ord_cols = list(ORDINAL)
    num_cols = [c for c in FEATURES + DERIVED if c not in ord_cols]
    prep = ColumnTransformer(
        [("ordinales", SimpleImputer(strategy="most_frequent"), ord_cols),
         ("numeriques", SimpleImputer(strategy="median"), num_cols)],
    ).set_output(transform="pandas")
    return Pipeline([("prep", prep), ("scaler", StandardScaler()),
                     ("modele", GradientBoostingRegressor(n_estimators=100, random_state=SEED))])
