# ======================================================================================
# Nom     : src/metrics.py
# Rôle    : Calcule les métriques d'évaluation des modèles (accuracy, précision, rappel, F1, ROC, AUC).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé pour calculer les indicateurs de performance.
# ======================================================================================

"""Métriques d'évaluation pour la classification binaire, from scratch.

Toutes les métriques sont calculées à partir de numpy uniquement, afin que
chaque formule soit visible et vérifiable. La convention est :
classe 1 = défaut de paiement (classe « positive »).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Métriques de base
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compte les vrais/faux positifs et négatifs (TP, TN, FP, FN) dans un
    dictionnaire. Le FN (défaut non détecté) est l'erreur la plus coûteuse pour
    la banque ; le FP (bon client refusé) est surtout un manque à gagner."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "TP": int(np.sum((y_true == 1) & (y_pred == 1))),
        "TN": int(np.sum((y_true == 0) & (y_pred == 0))),
        "FP": int(np.sum((y_true == 0) & (y_pred == 1))),
        "FN": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def accuracy(y_true, y_pred) -> float:
    """Exactitude : proportion de prédictions correctes, soit (TP + TN) / N."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred) -> float:
    """Précision : parmi les défauts prédits, proportion de vrais défauts, TP / (TP + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FP"]
    return cm["TP"] / denom if denom > 0 else 0.0


def recall(y_true, y_pred) -> float:
    """Rappel : parmi les vrais défauts, proportion détectée, TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FN"]
    return cm["TP"] / denom if denom > 0 else 0.0


def f1_score(y_true, y_pred) -> float:
    """F1-score, moyenne harmonique de la précision et du rappel. Critère de
    sélection des hyperparamètres : sur des classes déséquilibrées, l'accuracy
    seule serait trompeuse."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Courbe ROC et AUC
# ---------------------------------------------------------------------------

def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Calcule les points (fpr, tpr) de la courbe ROC pour tous les seuils.

    On trie les exemples par score décroissant : les sommes cumulées donnent
    directement le nombre de TP et de FP à chaque seuil, sans boucle sur les seuils.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    # Tri des exemples du score le plus élevé au plus faible.
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tps = np.cumsum(y_sorted)            # vrais positifs cumulés
    fps = np.cumsum(1 - y_sorted)        # faux positifs cumulés

    # On ne garde qu'un point par valeur de score distincte (les exemples
    # à score égal sont classés ensemble, quel que soit le seuil).
    distinct = np.where(np.diff(scores_sorted))[0]
    idx = np.concatenate([distinct, [len(y_sorted) - 1]])

    n_pos, n_neg = tps[-1], fps[-1]
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC indéfinie : une seule classe présente.")

    # Point de départ (0,0) : seuil au-dessus du score maximal.
    tpr = np.concatenate([[0.0], tps[idx] / n_pos])
    fpr = np.concatenate([[0.0], fps[idx] / n_neg])
    return fpr, tpr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Aire sous la courbe ROC (méthode des trapèzes) : 0,5 = hasard, 1 = parfait."""
    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# Regroupement des métriques
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true, y_pred, y_score=None) -> dict:
    """Calcule toutes les métriques et les regroupe dans un dictionnaire.
    L'AUC n'est ajoutée que si des scores continus (y_score) sont fournis."""
    result = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    if y_score is not None:
        fpr, tpr = roc_curve(y_true, y_score)
        result["auc_roc"] = auc(fpr, tpr)
    return result
