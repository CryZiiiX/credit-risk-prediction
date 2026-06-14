# ======================================================================================
# Nom     : src/preprocessing.py
# Rôle    : Contient les fonctions de prétraitement (nettoyage, encodage, imputation, normalisation) appliquées avant l'entraînement.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé par main.py lors de l'exécution du pipeline.
# ======================================================================================

"""Prétraitement des données : nettoyage, encodage, imputation, normalisation.

Anti-fuite de données : tout paramètre appris (médianes, moyennes/écarts-types)
est calculé sur le seul ensemble d'entraînement, puis appliqué tel quel à la
validation et au test.
"""

import numpy as np
import pandas as pd

from src.data_loader import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

# Encodage des variables catégorielles en entiers (mapping fixe).
# loan_grade est ordinale (A < ... < G) ; pour les variables nominales, ce
# label encoding reste acceptable pour l'arbre de décision et le k-NN.
ENCODING_MAPS = {
    "person_home_ownership": {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3},
    "loan_intent": {
        "PERSONAL": 0,
        "EDUCATION": 1,
        "MEDICAL": 2,
        "VENTURE": 3,
        "HOMEIMPROVEMENT": 4,
        "DEBTCONSOLIDATION": 5,
    },
    "loan_grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
    "cb_person_default_on_file": {"N": 0, "Y": 1},
}

# Seuils de nettoyage des valeurs aberrantes (âge > 120 ans, ancienneté d'emploi
# > 60 ans = erreurs de saisie ; voir data/stats/outliers_analysis.txt).
MAX_AGE = 100
MAX_EMP_LENGTH = 60


# ---------------------------------------------------------------------------
# Nettoyage et encodage
# ---------------------------------------------------------------------------


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les lignes physiquement impossibles (âge ou ancienneté d'emploi
    aberrants). Les valeurs extrêmes mais plausibles sont gardées : les modèles
    doivent apprendre la vraie distribution des données."""
    mask = (df["person_age"] <= MAX_AGE) & (
        df["person_emp_length"].isna() | (df["person_emp_length"] <= MAX_EMP_LENGTH)
    )
    return df[mask].reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les 4 variables catégorielles en entiers via un mapping fixe
    (ENCODING_MAPS), identique quel que soit le découpage. Une catégorie inconnue
    lève une erreur."""
    df = df.copy()
    for col, mapping in ENCODING_MAPS.items():
        unknown = set(df[col].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Catégories inconnues dans {col} : {unknown}")
        df[col] = df[col].map(mapping).astype(float)
    return df


# ---------------------------------------------------------------------------
# Imputation et normalisation
# ---------------------------------------------------------------------------


class MedianImputer:
    """Impute les valeurs manquantes par la médiane du train (robuste aux valeurs
    extrêmes, contrairement à la moyenne)."""

    def __init__(self):
        self.medians_ = None

    def fit(self, df: pd.DataFrame) -> "MedianImputer":
        """Apprend la médiane de chaque variable numérique (sur le train)."""
        self.medians_ = df[NUMERICAL_COLUMNS].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remplace les valeurs manquantes par les médianes apprises (celles du
        train, pour ne pas faire fuiter la validation ou le test)."""
        if self.medians_ is None:
            raise RuntimeError("Imputer non ajusté : appeler fit() d'abord.")
        df = df.copy()
        df[NUMERICAL_COLUMNS] = df[NUMERICAL_COLUMNS].fillna(self.medians_)
        return df


class StandardScaler:
    """Normalisation z-score (x - moyenne) / écart-type. Indispensable pour la
    régression logistique (convergence du gradient) et le k-NN (sinon les
    variables à grande échelle dominent la distance)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Apprend moyenne et écart-type de chaque variable (sur le train ; un
        écart-type nul est remplacé par 1 pour éviter une division par zéro)."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applique la normalisation z-score apprise par fit()."""
        if self.mean_ is None:
            raise RuntimeError("Scaler non ajusté : appeler fit() d'abord.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Enchaîne fit() puis transform() sur le même tableau."""
        return self.fit(X).transform(X)
