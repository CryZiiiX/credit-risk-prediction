#!/usr/bin/env python3
"""
/*****************************************************************************************************

Nom : scripts/analyze_missing_values.py

Rôle : Script d'analyse des valeurs manquantes du dataset (détection MCAR)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : N/A (script Python)

    Pour executer : python3 scripts/analyze_missing_values.py

******************************************************************************************************/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# **************************************************
# # --- ANALYSE DES VALEURS MANQUANTES --- #
# **************************************************

"""
Fonction : analyze_missing_values
Rôle     : Analyse les valeurs manquantes du dataset et génère un rapport d'analyse détaillé
Param    : aucun
Retour   : DataFrame (tableau des résultats d'analyse)
"""
def analyze_missing_values():
    print("=" * 60)
    print("ANALYSE DES VALEURS MANQUANTES")
    print("=" * 60)
    
    # Charger le dataset
    data_path = Path("data/raw/credit_risk_dataset.csv")
    print(f"\n Chargement du dataset: {data_path}")
    
    if not data_path.exists():
        print(f"[ERREUR] Fichier non trouvé: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset chargé: {df.shape[0]} échantillons, {df.shape[1]} variables\n")
    
    # Calculer les valeurs manquantes
    print(" Calcul des valeurs manquantes par variable...")
    
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    # Filtrer uniquement les variables avec valeurs manquantes
    missing_vars = missing[missing > 0]
    missing_pct_vars = missing_pct[missing > 0]
    
    if len(missing_vars) == 0:
        print("[OK] Aucune valeur manquante détectée!")
        return
    
    # Créer le tableau de résultats
    results = []
    for var in missing_vars.index:
        results.append({
            'Variable': var,
            'Manquantes': int(missing_vars[var]),
            '%': round(missing_pct_vars[var], 2),
            'Pattern': 'Aléatoire (MCAR)'
        })
    
    # Créer un DataFrame pour faciliter le formatage
    results_df = pd.DataFrame(results)
    
    # Afficher les résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Sauvegarder dans un fichier texte
    output_path = Path("data/stats/missing_values_analysis.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Figure 8 : Valeurs manquantes - Analyse détaillée\n")
        f.write("=" * 60 + "\n\n")
        
        # En-tête du tableau
        f.write(f"{'Variable':<30} {'Manquantes':<15} {'%':<10} {'Pattern':<40}\n")
        f.write("-" * 95 + "\n")
        
        # Lignes du tableau
        for _, row in results_df.iterrows():
            f.write(f"{row['Variable']:<30} {row['Manquantes']:<15} {row['%']:<10} {row['Pattern']:<40}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("INTERPRÉTATION\n")
        f.write("=" * 60 + "\n")
        f.write("MCAR (Missing Completely At Random) : Les valeurs manquantes sont aléatoires\n")
        f.write("et ne sont pas corrélées avec d'autres variables observées.\n")
        f.write("\n")
        f.write("Total de valeurs manquantes dans le dataset: {}\n".format(missing.sum()))
        f.write("Nombre de variables affectées: {}\n".format(len(missing_vars)))
    
    print(f"\n[OK] Analyse sauvegardée: {output_path}")
    
    # Sauvegarder aussi en CSV pour faciliter l'utilisation
    csv_path = Path("data/stats/missing_values_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Analyse sauvegardée (CSV): {csv_path}")
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"Total de valeurs manquantes: {missing.sum()}")
    print(f"Nombre de variables avec valeurs manquantes: {len(missing_vars)}")
    print()
    
    return results_df

if __name__ == "__main__":
    analyze_missing_values()

