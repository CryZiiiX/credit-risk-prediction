#!/usr/bin/env python3
# ======================================================================================
# Nom     : scripts/plot_sigmoid.py
# Rôle    : Trace la fonction sigmoïde (fonction d'activation de la régression logistique).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : python3 scripts/plot_sigmoid.py (depuis la racine du projet).
# ======================================================================================
"""Illustration de la fonction sigmoïde sigma(z) = 1 / (1 + e^(-z)).

Génère data/stats/sigmoïde_graphique.png. À lancer depuis la racine du projet.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATS_DIR = Path("data/stats")


def plot_sigmoid() -> None:
    """Trace la sigmoïde et l'enregistre dans data/stats/sigmoïde_graphique.png."""
    z = np.linspace(-10, 10, 400)
    sigma = 1.0 / (1.0 + np.exp(-z))

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, sigma, color="#2980b9", linewidth=2)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Fonction sigmoïde : sigma(z) = 1 / (1 + e^(-z))", fontweight="bold")
    ax.set_xlabel("z = w . x + b")
    ax.set_ylabel("sigma(z) (probabilité prédite)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(STATS_DIR / "sigmoïde_graphique.png", dpi=150)
    plt.close(fig)
    print("[OK] data/stats/sigmoïde_graphique.png")


if __name__ == "__main__":
    plot_sigmoid()
