#!/usr/bin/env python3
# ======================================================================================
# Nom     : scripts/detect_outliers.py
# Rôle    : Analyse des valeurs aberrantes du dataset (méthodes IQR et Z-score).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : python3 scripts/detect_outliers.py (depuis la racine du projet).
# ======================================================================================
"""Détection des valeurs aberrantes (IQR et Z-score) sur les variables numériques.

Écrit le rapport data/stats/outliers_analysis.txt. À lancer depuis la racine du projet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
STATS_DIR = Path("data/stats")
NUMERICAL_COLS = ["person_age", "person_income", "person_emp_length", "loan_amnt",
                  "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Masque booléen des aberrations par la règle IQR (Q1 - 1.5 IQR, Q3 + 1.5 IQR)."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)


def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
    """Masque booléen des aberrations par Z-score (|z| > threshold)."""
    mean, std = df[col].mean(), df[col].std()
    if std == 0:
        return pd.Series([False] * len(df), index=df.index)
    return np.abs((df[col] - mean) / std) > threshold


def analyze_outliers() -> None:
    """Analyse les aberrations des variables numériques et écrit outliers_analysis.txt."""
    print("=" * 60)
    print("ANALYSE DES VALEURS ABERRANTES")
    print("=" * 60)
    if not DATA_PATH.exists():
        print(f"[ERREUR] Fichier non trouvé : {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] {df.shape[0]} échantillons, {df.shape[1]} variables\n")

    cols = [c for c in NUMERICAL_COLS if c in df.columns]
    results = []
    for col in cols:
        if df[col].dropna().empty:
            continue
        iqr_mask = detect_outliers_iqr(df, col)
        z_mask = detect_outliers_zscore(df, col)
        res = {"Variable": col,
               "IQR_Count": int(iqr_mask.sum()),
               "IQR_Pct": round(100 * iqr_mask.sum() / len(df), 2),
               "ZScore_Count": int(z_mask.sum()),
               "ZScore_Pct": round(100 * z_mask.sum() / len(df), 2)}
        if iqr_mask.sum() > 0:
            vals = df.loc[iqr_mask, col]
            res.update(IQR_Min=vals.min(), IQR_Max=vals.max(), IQR_Mean=vals.mean(),
                       IQR_Indices=iqr_mask[iqr_mask].index.tolist()[:10])
        if z_mask.sum() > 0:
            vals = df.loc[z_mask, col]
            res.update(ZScore_Min=vals.min(), ZScore_Max=vals.max(),
                       ZScore_Mean=vals.mean(),
                       ZScore_Indices=z_mask[z_mask].index.tolist()[:10])
        results.append(res)
        print(f"{col}: IQR {res['IQR_Count']} ({res['IQR_Pct']}%), "
              f"Z-score {res['ZScore_Count']} ({res['ZScore_Pct']}%)")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out = STATS_DIR / "outliers_analysis.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("Figure : Détection des outliers - Analyse détaillée\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"Variable: {r['Variable']}\n")
            f.write("-" * 80 + "\n")
            f.write("Méthode IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR):\n")
            f.write(f"  Nombre d'outliers: {r['IQR_Count']}\n")
            f.write(f"  Pourcentage: {r['IQR_Pct']}%\n")
            if r["IQR_Count"] > 0:
                f.write(f"  Valeur min: {r['IQR_Min']:.2f}\n")
                f.write(f"  Valeur max: {r['IQR_Max']:.2f}\n")
                f.write(f"  Valeur moyenne: {r['IQR_Mean']:.2f}\n")
                f.write(f"  Exemples d'indices (max 10): {r['IQR_Indices']}\n")
            f.write("\nMéthode Z-score (|z| > 3):\n")
            f.write(f"  Nombre d'outliers: {r['ZScore_Count']}\n")
            f.write(f"  Pourcentage: {r['ZScore_Pct']}%\n")
            if r["ZScore_Count"] > 0:
                f.write(f"  Valeur min: {r['ZScore_Min']:.2f}\n")
                f.write(f"  Valeur max: {r['ZScore_Max']:.2f}\n")
                f.write(f"  Valeur moyenne: {r['ZScore_Mean']:.2f}\n")
                f.write(f"  Exemples d'indices (max 10): {r['ZScore_Indices']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        f.write("RÉSUMÉ GLOBAL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total d'outliers détectés (méthode IQR): {sum(r['IQR_Count'] for r in results)}\n")
        f.write(f"Total d'outliers détectés (méthode Z-score): {sum(r['ZScore_Count'] for r in results)}\n")
        f.write(f"Nombre de variables analysées: {len(results)}\n")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    analyze_outliers()
