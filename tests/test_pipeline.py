# ======================================================================================
# Nom     : tests/test_pipeline.py
# Rôle    : Contient les tests automatisés vérifiant le bon fonctionnement du pipeline.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Exécuter avec python3 -m pytest tests/ pour lancer les tests.
# ======================================================================================

"""Tests du pipeline : chargement, prétraitement, split, métriques, modèles.

Exécution :  python3 -m pytest tests/ -v

Les tests sur les modèles utilisent un petit jeu de données synthétique
linéairement séparable : tout modèle correct doit y obtenir une accuracy
quasi parfaite, ce qui valide l'implémentation sans dépendre du vrai dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, TARGET_COLUMN, load_raw_data
from src.metrics import (accuracy, auc, compute_all_metrics, confusion_matrix,
                         f1_score, precision, recall, roc_curve)
from src.models import DecisionTreeClassifier, KNNClassifier, LogisticRegression
from src.preprocessing import (MedianImputer, StandardScaler, clean_outliers,
                               encode_categoricals)
from src.split_data import stratified_split


# ---------------------------------------------------------------------------
# Données synthétiques
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """Crée un petit jeu de données synthétique (deux nuages gaussiens bien
    séparés) servant à tester les modèles indépendamment du vrai dataset."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-2.0, scale=0.5, size=(200, 3))
    X1 = rng.normal(loc=+2.0, scale=0.5, size=(200, 3))
    X = np.vstack([X0, X1])
    y = np.array([0] * 200 + [1] * 200)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle]


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def test_load_raw_data():
    """Vérifie que le dataset se charge avec les bonnes colonnes et une cible binaire."""
    df = load_raw_data()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 30000
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------

def test_preprocessing_produces_clean_numeric_data():
    """Vérifie que le prétraitement donne des données numériques, sans aberration
    ni valeur manquante."""
    df = encode_categoricals(clean_outliers(load_raw_data()))
    # Plus aucune valeur aberrante connue
    assert df["person_age"].max() <= 100
    # Toutes les colonnes sont numériques après encodage
    assert all(np.issubdtype(t, np.number) for t in df.dtypes)
    # L'imputation supprime toutes les valeurs manquantes
    imputed = MedianImputer().fit(df).transform(df)
    assert imputed.isna().sum().sum() == 0


def test_standard_scaler():
    """Vérifie que la normalisation produit des colonnes de moyenne 0 et
    d'écart-type 1."""
    rng = np.random.default_rng(1)
    X = rng.normal(loc=5.0, scale=3.0, size=(500, 4))
    Xs = StandardScaler().fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-9)


# ---------------------------------------------------------------------------
# Séparation des données
# ---------------------------------------------------------------------------

def test_stratified_split_proportions_and_no_overlap():
    """Vérifie les proportions 70/15/15, l'absence de chevauchement entre les
    ensembles et la conservation du taux de défaut (stratification)."""
    df = encode_categoricals(clean_outliers(load_raw_data()))
    df = df.reset_index(drop=True).reset_index(names="row_id")
    train, val, test = stratified_split(df, ratios=(0.70, 0.15, 0.15), seed=42)

    # Tailles : 70/15/15 à ±1 % près
    n = len(df)
    assert abs(len(train) / n - 0.70) < 0.01
    assert abs(len(val) / n - 0.15) < 0.01
    assert abs(len(test) / n - 0.15) < 0.01

    # Aucune ligne partagée entre deux ensembles (pas de fuite de données)
    ids = [set(s["row_id"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert ids[0] | ids[1] | ids[2] == set(df["row_id"])

    # Stratification : même taux de défaut partout (±1 point)
    rates = [s[TARGET_COLUMN].mean() for s in (train, val, test)]
    assert max(rates) - min(rates) < 0.01


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def test_metrics_on_known_example():
    """Vérifie les métriques sur un exemple dont les valeurs ont été calculées à
    la main."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    # TP=3, FN=1, FP=1, TN=5 → calculés à la main
    cm = confusion_matrix(y_true, y_pred)
    assert (cm["TP"], cm["FN"], cm["FP"], cm["TN"]) == (3, 1, 1, 5)
    assert accuracy(y_true, y_pred) == 0.8
    assert precision(y_true, y_pred) == 0.75
    assert recall(y_true, y_pred) == 0.75
    assert f1_score(y_true, y_pred) == 0.75


def test_roc_auc_perfect_and_random():
    """Vérifie que l'AUC vaut 1 pour un classement parfait et 0,5 quand tous les
    scores sont égaux (équivalent au hasard)."""
    y = np.array([0, 0, 0, 1, 1, 1])
    # Scores parfaitement ordonnés → AUC = 1
    fpr, tpr = roc_curve(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert auc(fpr, tpr) == pytest.approx(1.0)
    # Scores identiques pour tous → AUC = 0.5 (équivalent au hasard)
    fpr, tpr = roc_curve(y, np.full(6, 0.5))
    assert auc(fpr, tpr) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    LogisticRegression(learning_rate=0.5, n_iterations=500),
    DecisionTreeClassifier(max_depth=4, min_samples_split=5, min_samples_leaf=2),
    KNNClassifier(k=5),
])
def test_models_learn_separable_data(model, synthetic_data):
    """Vérifie que chaque modèle sépare presque parfaitement des données triviales
    et renvoie des probabilités valides (entre 0 et 1)."""
    X, y = synthetic_data
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    # Les probabilités sont bien dans [0, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # Données triviales → accuracy attendue quasi parfaite
    assert accuracy(y_test, model.predict(X_test)) >= 0.95


def test_logistic_regression_cost_decreases(synthetic_data):
    """Vérifie que la descente de gradient fait bien baisser le coût au fil des
    itérations."""
    X, y = synthetic_data
    lr = LogisticRegression(learning_rate=0.5, n_iterations=200).fit(X, y)
    # La descente de gradient doit faire baisser la log-loss
    assert lr.cost_history[-1] < lr.cost_history[0]


def test_compute_all_metrics_keys():
    """Vérifie que le dictionnaire renvoyé contient bien toutes les clés de
    métriques attendues."""
    y = np.array([0, 1, 0, 1])
    m = compute_all_metrics(y, y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert set(m) == {"accuracy", "precision", "recall", "f1_score",
                      "confusion_matrix", "auc_roc"}
