#!/usr/bin/env python3
"""
/*****************************************************************************************************

Nom : scripts/explore_data.py

Rôle : Script d'exploration du dataset Credit Risk (statistiques et visualisations)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : N/A (script Python)

    Pour executer : python3 scripts/explore_data.py

******************************************************************************************************/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# **************************************************
# # --- CHARGEMENT DES DONNÉES --- #
# **************************************************

"""
Fonction : load_data
Rôle     : Charge le dataset CSV depuis le répertoire data/raw
Param    : aucun
Retour   : DataFrame (dataset chargé)
"""
def load_data():
    data_path = Path("data/raw/credit_risk_dataset.csv")
    print(f" Chargement du dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    return df

# **************************************************
# # --- ANALYSE DES DONNÉES --- #
# **************************************************

"""
Fonction : print_basic_info
Rôle     : Affiche les informations de base du dataset (dimensions, types, statistiques)
Param    : df (DataFrame à analyser)
Retour   : void
"""
def print_basic_info(df):
    print("=" * 60)
    print("INFORMATIONS DE BASE")
    print("=" * 60)
    print(f"\nDimensions: {df.shape}")
    print(f"\nTypes de données:\n{df.dtypes}")
    print(f"\nPremières lignes:\n{df.head()}")
    print(f"\nStatistiques descriptives:\n{df.describe()}")
    print()

"""
Fonction : analyze_missing_values
Rôle     : Analyse et affiche les valeurs manquantes du dataset
Param    : df (DataFrame à analyser)
Retour   : void
"""
def analyze_missing_values(df):
    print("=" * 60)
    print("VALEURS MANQUANTES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    missing_table = pd.DataFrame({
        'Manquantes': missing,
        'Pourcentage': missing_pct
    })
    
    print("\n", missing_table[missing_table['Manquantes'] > 0])
    
    if missing.sum() == 0:
        print("\n[OK] Aucune valeur manquante détectée!")
    else:
        print(f"\n[WARNING] Total de valeurs manquantes: {missing.sum()}")
    print()

"""
Fonction : analyze_target_distribution
Rôle     : Analyse la distribution de la variable cible loan_status et génère une visualisation
Param    : df (DataFrame contenant les données)
Retour   : void
"""
def analyze_target_distribution(df):
    print("=" * 60)
    print("DISTRIBUTION DE LA VARIABLE CIBLE (loan_status)")
    print("=" * 60)
    
    target_counts = df['loan_status'].value_counts()
    target_pct = 100 * target_counts / len(df)
    
    print(f"\nClasse 0 (Pas de défaut): {target_counts[0]} ({target_pct[0]:.2f}%)")
    print(f"Classe 1 (Défaut):        {target_counts[1]} ({target_pct[1]:.2f}%)")
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Distribution de la Variable Cible (loan_status)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Statut du Prêt (0=Pas de défaut, 1=Défaut)')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_xticklabels(['Pas de défaut', 'Défaut'], rotation=0)
    
    # Ajouter les pourcentages
    for i, (count, pct) in enumerate(zip(target_counts, target_pct)):
        ax.text(i, count, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/stats/target_distribution.png', dpi=300)
    print("\n[OK] Graphique sauvegardé: data/stats/target_distribution.png")
    print()

"""
Fonction : analyze_numerical_features
Rôle     : Analyse la distribution des variables numériques et génère des histogrammes
Param    : df (DataFrame contenant les données)
Retour   : void
"""
def analyze_numerical_features(df):
    print("=" * 60)
    print("DISTRIBUTION DES VARIABLES NUMÉRIQUES")
    print("=" * 60)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('loan_status')  # Retirer la cible
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols[:9]):
        axes[idx].hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(col, fontweight='bold')
        axes[idx].set_xlabel('Valeur')
        axes[idx].set_ylabel('Fréquence')
        
        # Statistiques
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', label=f'Médiane: {median_val:.2f}')
        axes[idx].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('data/stats/numerical_distributions.png', dpi=300)
    print("\n[OK] Graphique sauvegardé: data/stats/numerical_distributions.png")
    print()

"""
Fonction : analyze_categorical_features
Rôle     : Analyse la distribution des variables catégorielles et génère des graphiques en barres
Param    : df (DataFrame contenant les données)
Retour   : void
"""
def analyze_categorical_features(df):
    print("=" * 60)
    print("DISTRIBUTION DES VARIABLES CATÉGORIELLES")
    print("=" * 60)
    
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 
                       'cb_person_default_on_file']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(categorical_cols):
        counts = df[col].value_counts()
        print(f"\n{col}:")
        print(counts)
        
        axes[idx].bar(range(len(counts)), counts.values, color='coral', edgecolor='black')
        axes[idx].set_title(col, fontweight='bold')
        axes[idx].set_xlabel('Catégorie')
        axes[idx].set_ylabel('Nombre')
        axes[idx].set_xticks(range(len(counts)))
        axes[idx].set_xticklabels(counts.index, rotation=45, ha='right')
        
        # Ajouter les valeurs
        for i, v in enumerate(counts.values):
            axes[idx].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/stats/categorical_distributions.png', dpi=300)
    print("\n[OK] Graphique sauvegardé: data/stats/categorical_distributions.png")
    print()

"""
Fonction : analyze_correlations
Rôle     : Calcule et visualise la matrice de corrélation entre variables numériques
Param    : df (DataFrame contenant les données)
Retour   : void
"""
def analyze_correlations(df):
    print("=" * 60)
    print("MATRICE DE CORRÉLATION")
    print("=" * 60)
    
    # Sélectionner les colonnes numériques
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculer la matrice de corrélation
    corr_matrix = df[numerical_cols].corr()
    
    # Visualisation
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de Corrélation des Variables Numériques', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('data/stats/correlation_matrix.png', dpi=300)
    
    print("\n[OK] Graphique sauvegardé: data/stats/correlation_matrix.png")
    
    # Afficher les corrélations fortes avec la cible
    print("\nCorrélations avec loan_status:")
    target_corr = corr_matrix['loan_status'].sort_values(ascending=False)
    print(target_corr)
    print()

"""
Fonction : analyze_by_target
Rôle     : Analyse les distributions des features numériques par classe cible et génère des boxplots
Param    : df (DataFrame contenant les données)
Retour   : void
"""
def analyze_by_target(df):
    print("=" * 60)
    print("ANALYSE PAR CLASSE CIBLE")
    print("=" * 60)
    
    numerical_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        df.boxplot(column=col, by='loan_status', ax=axes[idx])
        axes[idx].set_title(f'{col} par Statut du Prêt', fontweight='bold')
        axes[idx].set_xlabel('Statut (0=OK, 1=Défaut)')
        axes[idx].set_ylabel(col)
        plt.sca(axes[idx])
        plt.xticks([1, 2], ['Pas de défaut', 'Défaut'])
    
    plt.suptitle('')  # Supprimer le titre automatique de pandas
    plt.tight_layout()
    plt.savefig('data/stats/features_by_target.png', dpi=300)
    print("\n[OK] Graphique sauvegardé: data/stats/features_by_target.png")
    print()

"""
Fonction : save_summary
Rôle     : Sauvegarde un résumé textuel de l'exploration des données dans un fichier
Param    : df (DataFrame contenant les données)
Retour   : void
"""
def save_summary(df):
    output_path = Path("data/stats/data_exploration.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RÉSUMÉ DE L'EXPLORATION DES DONNÉES\n")
        f.write("Dataset: Credit Risk Prediction\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes\n\n")
        
        f.write("Variables:\n")
        for col in df.columns:
            f.write(f"  - {col} ({df[col].dtype})\n")
        
        f.write(f"\nValeurs manquantes: {df.isnull().sum().sum()}\n")
        
        f.write(f"\nBalance des classes:\n")
        f.write(f"  - Classe 0 (Pas de défaut): {(df['loan_status'] == 0).sum()} ({100*(df['loan_status'] == 0).sum()/len(df):.2f}%)\n")
        f.write(f"  - Classe 1 (Défaut): {(df['loan_status'] == 1).sum()} ({100*(df['loan_status'] == 1).sum()/len(df):.2f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("STATISTIQUES DESCRIPTIVES\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(df.describe()))
    
    print(f"[OK] Résumé sauvegardé: {output_path}\n")

"""
Fonction : main
Rôle     : Fonction principale orchestrant l'exploration complète du dataset
Param    : aucun
Retour   : void
"""
def main():
    print("\n" + "=" * 60)
    print("EXPLORATION DU DATASET CREDIT RISK")
    print("=" * 60 + "\n")
    
    # Créer le répertoire de stats si nécessaire
    Path("data/stats").mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    df = load_data()
    
    # Analyses
    print_basic_info(df)
    analyze_missing_values(df)
    analyze_target_distribution(df)
    analyze_numerical_features(df)
    analyze_categorical_features(df)
    analyze_correlations(df)
    analyze_by_target(df)
    save_summary(df)
    
    print("=" * 60)
    print("[OK] EXPLORATION TERMINÉE")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - data/stats/target_distribution.png")
    print("  - data/stats/numerical_distributions.png")
    print("  - data/stats/categorical_distributions.png")
    print("  - data/stats/correlation_matrix.png")
    print("  - data/stats/features_by_target.png")
    print("  - data/stats/data_exploration.txt")
    print()

if __name__ == "__main__":
    main()

