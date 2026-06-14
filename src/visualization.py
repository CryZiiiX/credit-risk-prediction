# ======================================================================================
# Nom     : src/visualization.py
# Rôle    : Génère les figures du rapport (matrices de confusion, courbes ROC, comparaison des modèles).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé par main.py pour produire les figures.
# ======================================================================================

"""Génération des figures du rapport (matplotlib uniquement).

Toutes les figures sont sauvegardées dans reports/figures/ en PNG.
Le backend 'Agg' permet de générer les images sans interface graphique
(utile en SSH, Docker ou intégration continue).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("reports/figures")


def _save(fig, name: str) -> Path:
    """Sauvegarde une figure en PNG dans reports/figures/ et la ferme."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(y_train, y_val, y_test) -> Path:
    """Trace le taux de défaut par ensemble (train/val/test) : vérifie que la
    stratification a conservé ~22 % de défauts partout."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sets = {"Train": y_train, "Validation": y_val, "Test": y_test}
    x = np.arange(len(sets))
    rate = [np.mean(y) * 100 for y in sets.values()]
    bars = ax.bar(x, rate, color="#c0392b", width=0.5)
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_xticks(x, sets.keys())
    ax.set_ylabel("Taux de défaut (%)")
    ax.set_title("Proportion de défauts par ensemble (split stratifié 70/15/15)")
    ax.set_ylim(0, 30)
    return _save(fig, "class_distribution.png")


def plot_confusion_matrix(cm: dict, model_name: str, filename: str) -> Path:
    """Trace la matrice de confusion 2×2 annotée d'un modèle (sur le test)."""
    matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        color = "white" if v > matrix.max() / 2 else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color=color, fontsize=14)
    ax.set_xticks([0, 1], ["Prédit : 0 (sain)", "Prédit : 1 (défaut)"])
    ax.set_yticks([0, 1], ["Réel : 0", "Réel : 1"])
    ax.set_title(f"Matrice de confusion (test) - {model_name}")
    fig.colorbar(im, shrink=0.8)
    return _save(fig, filename)


def plot_roc_curves(curves: dict) -> Path:
    """Trace les courbes ROC des modèles (curves = {nom: (fpr, tpr, auc)})."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (fpr, tpr, auc_value) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Hasard (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Courbes ROC - comparaison des modèles (test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, "roc_curves_comparison.png")


def plot_model_comparison(metrics: dict) -> Path:
    """Trace un diagramme en barres comparant les 5 métriques de test des modèles."""
    metric_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    labels = ["Accuracy", "Précision", "Rappel", "F1-score", "AUC-ROC"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model, m) in enumerate(metrics.items()):
        values = [m.get(name, 0) for name in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparaison des modèles sur l'ensemble de test")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "models_comparison.png")


def plot_cost_curve(cost_history: list) -> Path:
    """Trace le coût (log-loss) au fil des itérations : vérifie la convergence de
    la descente de gradient."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cost_history, color="#2980b9", linewidth=1.5)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Log-loss (entraînement)")
    ax.set_title("Convergence de la régression logistique")
    ax.grid(alpha=0.3)
    return _save(fig, "lr_cost_curve.png")
