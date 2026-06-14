# ======================================================================================
# Nom     : main.py
# Rôle    : Point d'entrée principal du pipeline expérimental (chargement, prétraitement, entraînement, évaluation et génération des figures).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Exécuter avec python3 main.py pour lancer le pipeline complet.
# ======================================================================================

"""Point d'entrée du pipeline de prédiction du risque de crédit.

Usage :
    python3 main.py                    # pipeline complet
    python3 main.py --compare-sklearn  # + vérification avec scikit-learn

Étapes :
    1. Chargement et nettoyage du dataset brut
    2. Encodage des variables catégorielles
    3. Split stratifié train (70 %) / validation (15 %) / test (15 %)
    4. Imputation et normalisation (paramètres appris sur le train uniquement)
    5. Entraînement des 3 modèles from scratch (LR, arbre CART, k-NN)
    6. Sélection des hyperparamètres sur la VALIDATION (F1-score)
    7. Évaluation finale sur le TEST (une seule fois)
    8. Sauvegarde : results/metrics.json, results/predictions.csv,
       results/models/*.pkl, reports/figures/*.png
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import TARGET_COLUMN, load_raw_data
from src.metrics import auc, compute_all_metrics, f1_score, roc_curve
from src.models import DecisionTreeClassifier, KNNClassifier, LogisticRegression
from src.preprocessing import (MedianImputer, StandardScaler, clean_outliers,
                               encode_categoricals)
from src.split_data import save_splits, stratified_split
from src import visualization as viz

RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"

# Grilles d'hyperparamètres explorées sur l'ensemble de validation.
# Volontairement petites : l'objectif est de montrer la démarche
# (sélection sur validation, jamais sur le test), pas une recherche exhaustive.
THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6]   # seuil de décision (LR et arbre)
DEPTH_GRID = [3, 5, 7, 9]               # profondeur de l'arbre
K_GRID = [5, 11, 21, 31]                # nombre de voisins du k-NN


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def to_xy(df: pd.DataFrame) -> tuple:
    """Sépare un DataFrame en matrice de variables X et vecteur cible y (0/1)."""
    X = df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    return X, y


def select_threshold(y_val, scores_val, grid=THRESHOLD_GRID) -> tuple:
    """Choisit dans `grid` le seuil qui maximise le F1 sur la validation
    (jamais sur le test, pour ne pas biaiser l'évaluation). Renvoie (seuil, F1)."""
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (scores_val >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def evaluate(name, y_train, scores_train, y_test, scores_test, threshold):
    """Calcule les métriques d'un modèle sur le train et le test, après application
    du seuil. Renvoie {"train": ..., "test": ...}."""
    return {
        "train": compute_all_metrics(
            y_train, (scores_train >= threshold).astype(int), scores_train),
        "test": compute_all_metrics(
            y_test, (scores_test >= threshold).astype(int), scores_test),
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(seed: int = 42, compare_sklearn: bool = False) -> dict:
    """Exécute tout le pipeline : chargement, prétraitement, découpage, entraînement
    des 3 modèles, sélection des hyperparamètres sur la validation, évaluation sur
    le test, puis sauvegarde des résultats et figures.

    Les hyperparamètres sont choisis sur la validation ; le test ne sert qu'une
    fois, à la fin, pour mesurer la performance réelle. Renvoie le dictionnaire des
    résultats (aussi écrit dans results/metrics.json).
    """
    t0 = time.time()

    print("[1/8] Chargement du dataset brut...")
    df = load_raw_data()
    n_raw = len(df)
    df = clean_outliers(df)
    print(f"      {n_raw} lignes chargées, {n_raw - len(df)} lignes aberrantes retirées.")

    print("[2/8] Encodage des variables catégorielles...")
    df = encode_categoricals(df)

    print("[3/8] Split stratifié train/validation/test (70/15/15)...")
    train, val, test = stratified_split(df, ratios=(0.70, 0.15, 0.15), seed=seed)
    save_splits(train, val, test)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)} "
          f"(taux de défaut : {train[TARGET_COLUMN].mean():.1%} / "
          f"{val[TARGET_COLUMN].mean():.1%} / {test[TARGET_COLUMN].mean():.1%})")

    print("[4/8] Imputation (médiane du train) et normalisation (z-score)...")
    imputer = MedianImputer().fit(train)
    train, val, test = (imputer.transform(d) for d in (train, val, test))

    X_train, y_train = to_xy(train)
    X_val, y_val = to_xy(val)
    X_test, y_test = to_xy(test)

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(X) for X in (X_train, X_val, X_test))

    all_metrics = {}
    test_curves = {}
    predictions = pd.DataFrame({"y_true": y_test})

    print("[5/8] Régression logistique : entraînement (descente de gradient)...")
    lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    lr.fit(X_train, y_train)
    t_lr, f1_lr = select_threshold(y_val, lr.predict_proba(X_val))
    print(f"      seuil retenu sur validation : {t_lr} (F1 val = {f1_lr:.3f})")

    scores = lr.predict_proba(X_test)
    all_metrics["logistic_regression"] = {
        "hyperparameters": {"learning_rate": 0.1, "n_iterations": 1000,
                            "threshold": t_lr},
        "validation_f1": f1_lr,
        **evaluate("lr", y_train, lr.predict_proba(X_train), y_test, scores, t_lr),
    }
    test_curves["Régression logistique"] = (*roc_curve(y_test, scores),)
    predictions["lr_proba"] = scores
    predictions["lr_pred"] = (scores >= t_lr).astype(int)

    print("[6/8] Arbre de décision CART : sélection de la profondeur sur validation...")
    best = {"f1": -1.0}
    for depth in DEPTH_GRID:
        tree = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
        t, f1 = select_threshold(y_val, tree.predict_proba(X_val))
        print(f"      max_depth={depth} -> F1 val = {f1:.3f} (seuil {t})")
        if f1 > best["f1"]:
            best = {"f1": f1, "depth": depth, "threshold": t, "model": tree}
    tree = best["model"]
    print(f"      retenu : max_depth={best['depth']}, seuil={best['threshold']} "
          f"({tree.n_nodes} nœuds)")

    scores = tree.predict_proba(X_test)
    all_metrics["decision_tree"] = {
        "hyperparameters": {"max_depth": best["depth"],
                            "min_samples_split": tree.min_samples_split,
                            "min_samples_leaf": tree.min_samples_leaf,
                            "threshold": best["threshold"],
                            "n_nodes": tree.n_nodes},
        "validation_f1": best["f1"],
        **evaluate("dt", y_train, tree.predict_proba(X_train), y_test, scores,
                   best["threshold"]),
    }
    test_curves["Arbre de décision"] = (*roc_curve(y_test, scores),)
    predictions["dt_proba"] = scores
    predictions["dt_pred"] = (scores >= best["threshold"]).astype(int)

    print("[7/8] k-NN : sélection de k sur validation...")
    knn = KNNClassifier().fit(X_train, y_train)
    best_knn = {"f1": -1.0}
    for k in K_GRID:
        t, f1 = select_threshold(y_val, knn.predict_proba(X_val, k=k))
        print(f"      k={k} -> F1 val = {f1:.3f} (seuil {t})")
        if f1 > best_knn["f1"]:
            best_knn = {"f1": f1, "k": k, "threshold": t}
    knn.k = best_knn["k"]
    print(f"      retenu : k={best_knn['k']}, seuil={best_knn['threshold']}")

    scores = knn.predict_proba(X_test)
    all_metrics["knn"] = {
        "hyperparameters": {"k": best_knn["k"], "threshold": best_knn["threshold"]},
        "validation_f1": best_knn["f1"],
        **evaluate("knn", y_train, knn.predict_proba(X_train), y_test, scores,
                   best_knn["threshold"]),
    }
    test_curves["k-NN"] = (*roc_curve(y_test, scores),)
    predictions["knn_proba"] = scores
    predictions["knn_pred"] = (scores >= best_knn["threshold"]).astype(int)

    print("[8/8] Sauvegarde des résultats et génération des figures...")
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    output = {
        "dataset": {"n_total": len(df), "n_train": len(train),
                    "n_val": len(val), "n_test": len(test),
                    "split": "70/15/15 stratifié", "seed": seed},
        "models": all_metrics,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    for name, model in [("logistic_regression", lr), ("decision_tree", tree),
                        ("knn", knn)]:
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    viz.plot_class_distribution(y_train, y_val, y_test)
    viz.plot_cost_curve(lr.cost_history)
    viz.plot_roc_curves({name: (fpr, tpr, auc(fpr, tpr))
                         for name, (fpr, tpr) in test_curves.items()})
    viz.plot_model_comparison({
        "Régression logistique": all_metrics["logistic_regression"]["test"],
        "Arbre de décision": all_metrics["decision_tree"]["test"],
        "k-NN": all_metrics["knn"]["test"],
    })
    for key, label, fname in [
        ("logistic_regression", "Régression logistique", "lr_confusion_matrix.png"),
        ("decision_tree", "Arbre de décision", "dt_confusion_matrix.png"),
        ("knn", "k-NN", "knn_confusion_matrix.png"),
    ]:
        viz.plot_confusion_matrix(all_metrics[key]["test"]["confusion_matrix"],
                                  label, fname)

    print(f"\n{'Modèle':<24}{'Accuracy':>10}{'Précision':>11}{'Rappel':>9}"
          f"{'F1':>8}{'AUC':>8}")
    for key, label in [("logistic_regression", "Régression logistique"),
                       ("decision_tree", "Arbre de décision"),
                       ("knn", "k-NN")]:
        m = all_metrics[key]["test"]
        print(f"{label:<24}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1_score']:>8.4f}{m['auc_roc']:>8.4f}")
    print(f"\nPipeline terminé en {time.time() - t0:.1f} s. "
          f"Résultats : results/metrics.json, figures : reports/figures/")

    if compare_sklearn:
        run_sklearn_comparison(X_train, y_train, X_test, y_test, all_metrics)

    return output


# ---------------------------------------------------------------------------
# Vérification avec scikit-learn
# ---------------------------------------------------------------------------

def run_sklearn_comparison(X_train, y_train, X_test, y_test, our_metrics):
    """Compare nos implémentations à celles de scikit-learn (mêmes données,
    hyperparamètres équivalents) et sauvegarde l'écart d'AUC dans
    results/sklearn_comparison.json. scikit-learn ne sert que de référence de
    vérification, pas de solution."""
    from sklearn.linear_model import LogisticRegression as SkLR
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier as SkDT

    print("\n--- Vérification avec scikit-learn (référence) ---")
    references = {
        "logistic_regression": SkLR(max_iter=1000),
        "decision_tree": SkDT(
            max_depth=our_metrics["decision_tree"]["hyperparameters"]["max_depth"],
            min_samples_split=20, min_samples_leaf=5, random_state=0),
        "knn": KNeighborsClassifier(
            n_neighbors=our_metrics["knn"]["hyperparameters"]["k"]),
    }
    comparison = {}
    for key, sk_model in references.items():
        sk_model.fit(X_train, y_train)
        sk_scores = sk_model.predict_proba(X_test)[:, 1]
        sk = compute_all_metrics(y_test, (sk_scores >= 0.5).astype(int), sk_scores)
        ours = our_metrics[key]["test"]
        comparison[key] = {"sklearn": sk,
                           "ecart_auc": round(ours["auc_roc"] - sk["auc_roc"], 4)}
        print(f"{key:<22} AUC from scratch = {ours['auc_roc']:.4f} | "
              f"AUC sklearn = {sk['auc_roc']:.4f} | "
              f"écart = {ours['auc_roc'] - sk['auc_roc']:+.4f}")

    with open(RESULTS_DIR / "sklearn_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print("Comparaison sauvegardée : results/sklearn_comparison.json")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de prédiction du risque de crédit (M1 TAA).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire du split (défaut : 42).")
    parser.add_argument("--compare-sklearn", action="store_true",
                        help="Vérifie les implémentations avec scikit-learn.")
    args = parser.parse_args()
    run_pipeline(seed=args.seed, compare_sklearn=args.compare_sklearn)
