# ======================================================================================
# Nom     : src/data_loader.py
# Rôle    : Charge et valide le jeu de données brut utilisé par le pipeline.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé par main.py lors de l'exécution du pipeline.
# ======================================================================================

"""Chargement et validation du dataset brut.

Le dataset (Kaggle - Credit Risk Dataset) contient 32 581 emprunteurs décrits
par 11 variables explicatives (8 numériques, 3+1 catégorielles) et une cible
binaire `loan_status` (1 = défaut de paiement, 0 = remboursement normal).
"""

from pathlib import Path

import pandas as pd

# Chemin par défaut du CSV brut, relatif à la racine du projet.
RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")

# Colonnes attendues dans le fichier brut. Si le fichier ne les contient pas
# toutes, on préfère échouer immédiatement avec un message clair plutôt que
# de produire des résultats faux en silence.
EXPECTED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

TARGET_COLUMN = "loan_status"

CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

NUMERICAL_COLUMNS = [c for c in EXPECTED_COLUMNS
                     if c not in CATEGORICAL_COLUMNS and c != TARGET_COLUMN]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut et vérifie sa structure (colonnes attendues, non vide).
    Lève une erreur explicite en cas de problème plutôt que de continuer avec des
    données invalides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}. "
            "Vérifiez que data/raw/credit_risk_dataset.csv existe."
        )

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

    if df.empty:
        raise ValueError("Le dataset est vide.")

    return df[EXPECTED_COLUMNS]
