#!/usr/bin/env python3
# ======================================================================================
# Nom     : scripts/explore_data.py
# Rôle    : Analyse exploratoire du dataset Credit Risk (statistiques et figures EDA).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : python3 scripts/explore_data.py (depuis la racine du projet).
# ======================================================================================
"""Analyse exploratoire des données (EDA) : distributions, corrélations, résumé.

Génère dans data/stats/ : target_distribution.png, numerical_distributions.png,
categorical_distributions.png, correlation_matrix.png, features_by_target.png,
default_rate_by_grade.txt/.csv et data_exploration.txt. À lancer depuis la racine
du projet.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
STATS_DIR = Path("data/stats")


def load_data() -> pd.DataFrame:
    """Charge le dataset brut depuis data/raw/."""
    print(f"Chargement du dataset : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    return df


def analyze_target_distribution(df: pd.DataFrame) -> None:
    """Trace la distribution de la cible loan_status -> target_distribution.png."""
    target_counts = df["loan_status"].value_counts()
    target_pct = 100 * target_counts / len(df)
    print(f"Classe 0 (pas de défaut) : {target_counts[0]} ({target_pct[0]:.2f}%)")
    print(f"Classe 1 (défaut)        : {target_counts[1]} ({target_pct[1]:.2f}%)")

    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Distribution de la variable cible (loan_status)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Statut du prêt (0 = pas de défaut, 1 = défaut)")
    ax.set_ylabel("Nombre d'échantillons")
    ax.set_xticklabels(["Pas de défaut", "Défaut"], rotation=0)
    for i, (count, pct) in enumerate(zip(target_counts, target_pct)):
        ax.text(i, count, f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(STATS_DIR / "target_distribution.png", dpi=300)
    plt.close(fig)
    print("[OK] data/stats/target_distribution.png\n")


def analyze_numerical_features(df: pd.DataFrame) -> None:
    """Histogrammes des variables numériques -> numerical_distributions.png."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove("loan_status")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for idx, col in enumerate(numerical_cols[:9]):
        axes[idx].hist(df[col].dropna(), bins=50, color="steelblue",
                       edgecolor="black", alpha=0.7)
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].set_xlabel("Valeur")
        axes[idx].set_ylabel("Fréquence")
        mean_val, median_val = df[col].mean(), df[col].median()
        axes[idx].axvline(mean_val, color="red", linestyle="--",
                          label=f"Moyenne : {mean_val:.2f}")
        axes[idx].axvline(median_val, color="green", linestyle="--",
                          label=f"Médiane : {median_val:.2f}")
        axes[idx].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(STATS_DIR / "numerical_distributions.png", dpi=300)
    plt.close(fig)
    print("[OK] data/stats/numerical_distributions.png\n")


def analyze_categorical_features(df: pd.DataFrame) -> None:
    """Diagrammes en barres des variables catégorielles -> categorical_distributions.png."""
    categorical_cols = ["person_home_ownership", "loan_intent", "loan_grade",
                        "cb_person_default_on_file"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for idx, col in enumerate(categorical_cols):
        counts = df[col].value_counts()
        axes[idx].bar(range(len(counts)), counts.values, color="coral",
                      edgecolor="black")
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].set_xlabel("Catégorie")
        axes[idx].set_ylabel("Nombre")
        axes[idx].set_xticks(range(len(counts)))
        axes[idx].set_xticklabels(counts.index, rotation=45, ha="right")
        for i, v in enumerate(counts.values):
            axes[idx].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(STATS_DIR / "categorical_distributions.png", dpi=300)
    plt.close(fig)
    print("[OK] data/stats/categorical_distributions.png\n")


def analyze_correlations(df: pd.DataFrame) -> None:
    """Matrice de corrélation des variables numériques -> correlation_matrix.png."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Matrice de corrélation des variables numériques",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(STATS_DIR / "correlation_matrix.png", dpi=300)
    plt.close(fig)
    print("[OK] data/stats/correlation_matrix.png")
    print("Corrélations avec loan_status :")
    print(corr_matrix["loan_status"].sort_values(ascending=False), "\n")


def analyze_by_target(df: pd.DataFrame) -> None:
    """Boxplots des features numériques par classe cible -> features_by_target.png."""
    numerical_cols = ["person_age", "person_income", "loan_amnt", "loan_int_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for idx, col in enumerate(numerical_cols):
        df.boxplot(column=col, by="loan_status", ax=axes[idx])
        axes[idx].set_title(f"{col} par statut du prêt", fontweight="bold")
        axes[idx].set_xlabel("Statut (0 = OK, 1 = défaut)")
        axes[idx].set_ylabel(col)
        plt.sca(axes[idx])
        plt.xticks([1, 2], ["Pas de défaut", "Défaut"])
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(STATS_DIR / "features_by_target.png", dpi=300)
    plt.close(fig)
    print("[OK] data/stats/features_by_target.png\n")


def analyze_default_rate_by_grade(df: pd.DataFrame) -> None:
    """Taux de défaut par loan_grade -> default_rate_by_grade.txt et .csv."""
    rows = []
    for grade in sorted(df["loan_grade"].unique()):
        sub = df[df["loan_grade"] == grade]
        defauts = int((sub["loan_status"] == 1).sum())
        rows.append({"Loan Grade": grade, "Total": len(sub),
                     "Pas de défaut": len(sub) - defauts, "Défauts": defauts,
                     "Taux défaut (%)": 100 * defauts / len(sub)})
    res = pd.DataFrame(rows)
    res.to_csv(STATS_DIR / "default_rate_by_grade.csv", index=False)
    with open(STATS_DIR / "default_rate_by_grade.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("TAUX DE DÉFAUT PAR CATÉGORIE LOAN_GRADE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(df)} échantillons\n\n")
        f.write(res.to_string(index=False) + "\n\n")
        f.write("=" * 60 + "\n")
        f.write("INTERPRÉTATION\n")
        f.write("=" * 60 + "\n")
        f.write("Le taux de défaut augmente généralement avec la dégradation du grade.\n")
        f.write("Grade A = meilleur risque, Grade G = pire risque.\n\n")
        f.write("Observations:\n")
        for _, r in res.iterrows():
            f.write(f"  - Grade {r['Loan Grade']}: {r['Taux défaut (%)']:.2f}% de défaut "
                    f"({r['Défauts']}/{r['Total']} échantillons)\n")
    print("[OK] data/stats/default_rate_by_grade.txt + .csv\n")


def save_summary(df: pd.DataFrame) -> None:
    """Écrit un résumé textuel de l'exploration -> data_exploration.txt."""
    with open(STATS_DIR / "data_exploration.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RÉSUMÉ DE L'EXPLORATION DES DONNÉES\n")
        f.write("Dataset: Credit Risk Prediction\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes\n\n")
        f.write("Variables:\n")
        for col in df.columns:
            f.write(f"  - {col} ({df[col].dtype})\n")
        f.write(f"\nValeurs manquantes: {df.isnull().sum().sum()}\n")
        f.write("\nBalance des classes:\n")
        n0 = int((df["loan_status"] == 0).sum())
        n1 = int((df["loan_status"] == 1).sum())
        f.write(f"  - Classe 0 (Pas de défaut): {n0} ({100 * n0 / len(df):.2f}%)\n")
        f.write(f"  - Classe 1 (Défaut): {n1} ({100 * n1 / len(df):.2f}%)\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("STATISTIQUES DESCRIPTIVES\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(df.describe()))
    print("[OK] data/stats/data_exploration.txt\n")


def main() -> None:
    """Lance l'exploration complète et génère les 6 sorties EDA dans data/stats/."""
    print("=" * 60)
    print("EXPLORATION DU DATASET CREDIT RISK")
    print("=" * 60 + "\n")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    analyze_target_distribution(df)
    analyze_numerical_features(df)
    analyze_categorical_features(df)
    analyze_correlations(df)
    analyze_by_target(df)
    analyze_default_rate_by_grade(df)
    save_summary(df)
    print("[OK] Exploration terminée. Fichiers générés dans data/stats/.")


if __name__ == "__main__":
    main()
