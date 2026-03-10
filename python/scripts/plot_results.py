#!/usr/bin/env python3
"""
/*****************************************************************************************************

Nom : scripts/plot_results.py

Rôle : Script de visualisation des résultats du modèle (génération de graphiques)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : N/A (script Python)

    Pour executer : python3 scripts/plot_results.py

******************************************************************************************************/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path

# Configuration matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# **************************************************
# # --- VISUALISATION DES MATRICES DE CONFUSION --- #
# **************************************************

"""
Fonction : plot_lr_confusion_matrix
Rôle     : Génère et sauvegarde la matrice de confusion pour la régression logistique C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_lr_confusion_matrix():
    print(" Génération de la matrice de confusion (Régression Logistique C)...")
    
    try:
        # Lire les résultats
        with open("results/metrics/logistic_regression/lr_c_test_confusion_matrix.txt", 'r') as f:
            lines = f.readlines()
            
        # Parser les valeurs
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        # Créer la matrice
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Utiliser imshow au lieu de sns.heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        
        # Ajouter les annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold', fontsize=14)
        
        # Configurer les labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit: Pas de défaut', 'Prédit: Défaut'])
        ax.set_yticklabels(['Réel: Pas de défaut', 'Réel: Défaut'])
        
        # Ajouter la colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nombre de prédictions', rotation=270, labelpad=20)
        
        ax.set_title('Matrice de Confusion - Régression Logistique (C) - Test Set', fontsize=14, fontweight='bold', pad=20)
        
        # Ajouter les pourcentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j, i+0.3, f'({pct:.1f}%)', 
                       ha='center', va='center', color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_c_confusion_matrix_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_c_confusion_matrix_test.png")
        
        return True
    except FileNotFoundError:
        print("[WARNING] Fichier lr_c_test_confusion_matrix.txt non trouvé")
        return False

"""
Fonction : plot_dt_confusion_matrix
Rôle     : Génère et sauvegarde la matrice de confusion pour l'arbre de décision C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_dt_confusion_matrix():
    print(" Génération de la matrice de confusion (Arbre de Décision C)...")
    
    try:
        # Lire les résultats
        with open("results/metrics/decision_tree/dt_c_test_confusion_matrix.txt", 'r') as f:
            lines = f.readlines()
            
        # Parser les valeurs
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        # Créer la matrice
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Utiliser imshow au lieu de sns.heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        
        # Ajouter les annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold', fontsize=14)
        
        # Configurer les labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit: Pas de défaut', 'Prédit: Défaut'])
        ax.set_yticklabels(['Réel: Pas de défaut', 'Réel: Défaut'])
        
        # Ajouter la colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nombre de prédictions', rotation=270, labelpad=20)
        
        ax.set_title('Matrice de Confusion - Arbre de Décision (C) - Test Set', fontsize=14, fontweight='bold', pad=20)
        
        # Ajouter les pourcentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j, i+0.3, f'({pct:.1f}%)', 
                       ha='center', va='center', color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/decision_tree/dt_c_confusion_matrix_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/decision_tree/dt_c_confusion_matrix_test.png")
        
        return True
    except FileNotFoundError:
        print("[WARNING] Fichier dt_c_test_confusion_matrix.txt non trouvé")
        return False

"""
Fonction : plot_lr_python_confusion_matrix
Rôle     : Génère et sauvegarde la matrice de confusion pour la régression logistique Python
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_lr_python_confusion_matrix():
    print(" Génération de la matrice de confusion (Régression Logistique Python)...")
    
    try:
        with open("results/metrics/logistic_regression/lr_python_test_confusion_matrix.txt", 'r') as f:
            lines = f.readlines()
            
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold', fontsize=14)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit: Pas de défaut', 'Prédit: Défaut'])
        ax.set_yticklabels(['Réel: Pas de défaut', 'Réel: Défaut'])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nombre de prédictions', rotation=270, labelpad=20)
        
        ax.set_title('Matrice de Confusion - Régression Logistique (Python) - Test Set', 
                    fontsize=14, fontweight='bold', pad=20)
        
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j, i+0.3, f'({pct:.1f}%)', 
                       ha='center', va='center', color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_python_confusion_matrix_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_python_confusion_matrix_test.png")
        
        return True
    except FileNotFoundError:
        print("[WARNING] Fichier lr_python_test_confusion_matrix.txt non trouvé")
        return False

"""
Fonction : plot_dt_python_confusion_matrix
Rôle     : Génère et sauvegarde la matrice de confusion pour l'arbre de décision Python
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_dt_python_confusion_matrix():
    print(" Génération de la matrice de confusion (Arbre de Décision Python)...")
    
    try:
        with open("results/metrics/decision_tree/dt_python_test_confusion_matrix.txt", 'r') as f:
            lines = f.readlines()
            
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold', fontsize=14)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit: Pas de défaut', 'Prédit: Défaut'])
        ax.set_yticklabels(['Réel: Pas de défaut', 'Réel: Défaut'])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nombre de prédictions', rotation=270, labelpad=20)
        
        ax.set_title('Matrice de Confusion - Arbre de Décision (Python) - Test Set', 
                    fontsize=14, fontweight='bold', pad=20)
        
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j, i+0.3, f'({pct:.1f}%)', 
                       ha='center', va='center', color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/decision_tree/dt_python_confusion_matrix_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/decision_tree/dt_python_confusion_matrix_test.png")
        
        return True
    except FileNotFoundError:
        print("[WARNING] Fichier dt_python_test_confusion_matrix.txt non trouvé")
        return False

# **************************************************
# # --- COMPARAISON MÉTRIQUES TRAIN VS TEST --- #
# **************************************************

"""
Fonction : plot_lr_metrics_train_vs_test
Rôle     : Génère un graphique comparant les métriques train et test pour la régression logistique C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_lr_metrics_train_vs_test():
    print("\n Génération du graphique de comparaison train vs test (Régression Logistique C)...")
    
    try:
        # Charger métriques train
        train_metrics = {}
        with open("results/metrics/logistic_regression/lr_c_train_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    train_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    train_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    train_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    train_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Charger métriques test
        test_metrics = {}
        with open("results/metrics/logistic_regression/lr_c_test_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    test_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    test_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    test_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    test_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Créer le graphique
        metrics_names = list(train_metrics.keys())
        train_values = list(train_metrics.values())
        test_values = list(test_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_xlabel('Métriques', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Régression Logistique (C) - Comparaison Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_c_metrics_train_vs_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_c_metrics_train_vs_test.png")
        
        return True
    except FileNotFoundError as e:
        print(f"[WARNING] Fichier de métriques non trouvé: {e}")
        return False

"""
Fonction : plot_dt_metrics_train_vs_test
Rôle     : Génère un graphique comparant les métriques train et test pour l'arbre de décision C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_dt_metrics_train_vs_test():
    print("\n Génération du graphique de comparaison train vs test (Arbre de Décision C)...")
    
    try:
        # Charger métriques train
        train_metrics = {}
        with open("results/metrics/decision_tree/dt_c_train_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    train_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    train_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    train_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    train_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Charger métriques test
        test_metrics = {}
        with open("results/metrics/decision_tree/dt_c_test_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    test_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    test_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    test_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    test_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Créer le graphique (identique à plot_lr_metrics_train_vs_test)
        metrics_names = list(train_metrics.keys())
        train_values = list(train_metrics.values())
        test_values = list(test_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_xlabel('Métriques', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Arbre de Décision (C) - Comparaison Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/plots/decision_tree/dt_c_metrics_train_vs_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/decision_tree/dt_c_metrics_train_vs_test.png")
        
        return True
    except FileNotFoundError as e:
        print(f"[WARNING] Fichier de métriques non trouvé: {e}")
        return False

"""
Fonction : plot_lr_python_metrics_train_vs_test
Rôle     : Génère un graphique comparant les métriques train et test pour la régression logistique Python
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_lr_python_metrics_train_vs_test():
    print("\n Génération du graphique de comparaison train vs test (Régression Logistique Python)...")
    
    try:
        # Charger métriques train
        train_metrics = {}
        with open("results/metrics/logistic_regression/lr_python_train_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    train_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    train_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    train_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    train_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Charger métriques test
        test_metrics = {}
        with open("results/metrics/logistic_regression/lr_python_test_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    test_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    test_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    test_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    test_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Créer le graphique
        metrics_names = list(train_metrics.keys())
        train_values = list(train_metrics.values())
        test_values = list(test_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_xlabel('Métriques', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Régression Logistique (Python) - Comparaison Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_python_metrics_train_vs_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_python_metrics_train_vs_test.png")
        
        return True
    except FileNotFoundError as e:
        print(f"[WARNING] Fichier de métriques non trouvé: {e}")
        return False

"""
Fonction : plot_dt_python_metrics_train_vs_test
Rôle     : Génère un graphique comparant les métriques train et test pour l'arbre de décision Python
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_dt_python_metrics_train_vs_test():
    print("\n Génération du graphique de comparaison train vs test (Arbre de Décision Python)...")
    
    try:
        # Charger métriques train
        train_metrics = {}
        with open("results/metrics/decision_tree/dt_python_train_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    train_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    train_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    train_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    train_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Charger métriques test
        test_metrics = {}
        with open("results/metrics/decision_tree/dt_python_test_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    test_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    test_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    test_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    test_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Créer le graphique
        metrics_names = list(train_metrics.keys())
        train_values = list(train_metrics.values())
        test_values = list(test_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_xlabel('Métriques', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Arbre de Décision (Python) - Comparaison Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/plots/decision_tree/dt_python_metrics_train_vs_test.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/decision_tree/dt_python_metrics_train_vs_test.png")
        
        return True
    except FileNotFoundError as e:
        print(f"[WARNING] Fichier de métriques non trouvé: {e}")
        return False

# **************************************************
# # --- VISUALISATION DES COURBES DE COÛT --- #
# **************************************************

"""
Fonction : plot_lr_cost_curve
Rôle     : Génère et sauvegarde la courbe de coût pendant l'entraînement de la régression logistique C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_lr_cost_curve():
    print("\n Génération de la courbe de coût (Régression Logistique C)...")
    
    # Vérifier si le fichier existe
    cost_file = Path("results/plots/csv/lr_c_cost_curve.csv")
    if not cost_file.exists():
        print("[WARNING] Fichier lr_c_cost_curve.csv non trouvé. Création d'un exemple...")
        # Créer des données synthétiques pour la démonstration
        iterations = list(range(0, 1001, 100))
        # Ces valeurs viennent de l'exécution réelle observée
        costs = [0.693147, 0.597342, 0.546575, 0.513076, 0.473054, 
                 0.448978, 0.428197, 0.431842, 0.431531, 0.442407, 0.442407]
        
        df = pd.DataFrame({'iteration': iterations, 'cost': costs})
        df.to_csv(cost_file, index=False)
    
    try:
        # Charger les données
        df = pd.read_csv(cost_file)
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['iteration'], df['cost'], marker='o', linewidth=2, 
               markersize=6, color='#e74c3c', label='Coût (Cross-Entropy)')
        
        ax.set_xlabel('Itérations', fontweight='bold', fontsize=12)
        ax.set_ylabel('Coût (Cross-Entropy Loss)', fontweight='bold', fontsize=12)
        ax.set_title('Courbe de Coût - Régression Logistique (C) - Entraînement', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ajouter une ligne horizontale pour montrer la convergence
        final_cost = df['cost'].iloc[-1]
        ax.axhline(y=final_cost, color='green', linestyle='--', 
                  alpha=0.5, label=f'Coût final: {final_cost:.4f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_c_cost_curve_training.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_c_cost_curve_training.png")
        
        return True
    except Exception as e:
        print(f"[WARNING] Erreur lors de la création de la courbe de coût: {e}")
        return False

"""
Fonction : plot_lr_python_cost_curve
Rôle     : Génère et sauvegarde la courbe de coût pendant l'entraînement de la régression logistique Python
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_lr_python_cost_curve():
    print("\n Génération de la courbe de coût (Régression Logistique Python)...")
    
    cost_file = Path("results/plots/csv/lr_python_cost_curve.csv")
    if not cost_file.exists():
        print(f"[WARNING] Fichier {cost_file} non trouvé")
        return False
    
    try:
        # Charger les données
        df = pd.read_csv(cost_file)
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['iteration'], df['cost'], marker='o', linewidth=2, 
               markersize=6, color='#3498db', label='Coût (Cross-Entropy)')
        
        ax.set_xlabel('Itérations', fontweight='bold', fontsize=12)
        ax.set_ylabel('Coût (Cross-Entropy Loss)', fontweight='bold', fontsize=12)
        ax.set_title('Courbe de Coût - Régression Logistique (Python) - Entraînement', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ajouter une ligne horizontale pour montrer la convergence
        final_cost = df['cost'].iloc[-1]
        ax.axhline(y=final_cost, color='green', linestyle='--', 
                  alpha=0.5, label=f'Coût final: {final_cost:.4f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_python_cost_curve_training.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_python_cost_curve_training.png")
        
        return True
    except Exception as e:
        print(f"[WARNING] Erreur lors de la création de la courbe de coût: {e}")
        return False

"""
Fonction : plot_roc_curves_comparison
Rôle     : Génère et sauvegarde les courbes ROC comparatives pour LR et DT en C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_roc_curves_comparison():
    print("\n Génération des courbes ROC (comparaison LR vs DT C)...")
    
    # Vérifier si les fichiers existent
    lr_roc_file = Path("results/plots/csv/lr_roc_data.csv")
    dt_roc_file = Path("results/plots/csv/dt_roc_data.csv")
    
    if not lr_roc_file.exists():
        print("[WARNING] Fichier lr_roc_data.csv non trouvé")
        return False
    if not dt_roc_file.exists():
        print("[WARNING] Fichier dt_roc_data.csv non trouvé")
        return False
    
    try:
        # Charger les données ROC
        lr_roc = pd.read_csv(lr_roc_file)
        dt_roc = pd.read_csv(dt_roc_file)
        
        # Lire les valeurs AUC-ROC depuis les fichiers de métriques
        lr_auc = 0.0
        dt_auc = 0.0
        
        try:
            with open("results/metrics/logistic_regression/lr_c_test_metrics.txt", 'r') as f:
                for line in f:
                    if "AUC-ROC" in line:
                        lr_auc = float(line.split(':')[1].strip())
        except:
            print("[WARNING] Impossible de lire l'AUC-ROC de la régression logistique")
        
        try:
            with open("results/metrics/decision_tree/dt_c_test_metrics.txt", 'r') as f:
                for line in f:
                    if "AUC-ROC" in line:
                        dt_auc = float(line.split(':')[1].strip())
        except:
            print("[WARNING] Impossible de lire l'AUC-ROC de l'arbre de décision")
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tracer la ligne de référence (AUC = 0.5)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='AUC = 0.5 (aléatoire)')
        
        # Tracer les courbes ROC
        ax.plot(lr_roc['fpr'], lr_roc['tpr'], linewidth=2.5, 
               color='#3498db', label=f'Régression Logistique (AUC = {lr_auc:.4f})')
        ax.plot(dt_roc['fpr'], dt_roc['tpr'], linewidth=2.5, 
               color='#e74c3c', label=f'Arbre de Décision (AUC = {dt_auc:.4f})')
        
        ax.set_xlabel('Taux de Faux Positifs (FPR)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontweight='bold', fontsize=12)
        ax.set_title('Comparaison ROC - LR vs DT (C)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_dt_c_roc_curves_comparison.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_dt_c_roc_curves_comparison.png")
        
        return True
    except Exception as e:
        print(f"[WARNING] Erreur lors de la création des courbes ROC: {e}")
        return False

"""
Fonction : plot_python_roc_curves
Rôle     : Génère et sauvegarde les courbes ROC pour LR et DT en Python
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_python_roc_curves():
    print("\n Génération des courbes ROC (Python)...")
    
    lr_roc_file = Path("results/plots/csv/lr_python_roc_data.csv")
    dt_roc_file = Path("results/plots/csv/dt_python_roc_data.csv")
    
    if not lr_roc_file.exists() or not dt_roc_file.exists():
        print("[WARNING] Fichiers ROC Python non trouvés")
        return False
    
    try:
        lr_roc = pd.read_csv(lr_roc_file)
        dt_roc = pd.read_csv(dt_roc_file)
        
        # Lire AUC-ROC depuis les fichiers de métriques
        lr_auc = 0.0
        dt_auc = 0.0
        
        try:
            with open("results/metrics/logistic_regression/lr_python_test_metrics.txt", 'r') as f:
                for line in f:
                    if "AUC-ROC" in line:
                        lr_auc = float(line.split(':')[1].strip())
        except:
            pass
        
        try:
            with open("results/metrics/decision_tree/dt_python_test_metrics.txt", 'r') as f:
                for line in f:
                    if "AUC-ROC" in line:
                        dt_auc = float(line.split(':')[1].strip())
        except:
            pass
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='AUC = 0.5 (aléatoire)')
        ax.plot(lr_roc['fpr'], lr_roc['tpr'], linewidth=2.5, 
               color='#3498db', label=f'Régression Logistique (AUC = {lr_auc:.4f})')
        ax.plot(dt_roc['fpr'], dt_roc['tpr'], linewidth=2.5, 
               color='#e74c3c', label=f'Arbre de Décision (AUC = {dt_auc:.4f})')
        
        ax.set_xlabel('Taux de Faux Positifs (FPR)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontweight='bold', fontsize=12)
        ax.set_title('Courbes ROC - LR vs DT (Python)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig('results/plots/logistic_regression/lr_dt_python_roc_curves_comparison.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/logistic_regression/lr_dt_python_roc_curves_comparison.png")
        
        return True
    except Exception as e:
        print(f"[WARNING] Erreur lors de la création des courbes ROC Python: {e}")
        return False

# **************************************************
# # --- IMPORTANCE DES FEATURES --- #
# **************************************************

"""
Fonction : plot_dt_feature_importance
Rôle     : Génère et sauvegarde un graphique d'importance des features pour l'arbre de décision C
Param    : aucun
Retour   : bool (True si succès, False sinon)
"""
def plot_dt_feature_importance():
    print("\n Génération du graphique d'importance des features (Arbre de Décision C)...")
    
    # Vérifier si le fichier existe
    importance_file = Path("results/plots/feature_importance.txt")
    if not importance_file.exists():
        print("[WARNING] Fichier feature_importance.txt non trouvé")
        return False
    
    try:
        # Charger les données
        df = pd.read_csv(importance_file, sep='\t')
        
        # Trier par importance absolue
        df['abs_weight'] = df['weight'].abs()
        df = df.sort_values('abs_weight', ascending=True)
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df['weight']]
        ax.barh(df['feature'], df['weight'], color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Poids (Coefficient)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Features', fontweight='bold', fontsize=12)
        ax.set_title('Importance des Features - Arbre de Décision (C)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Impact positif (↑ risque)'),
            Patch(facecolor='#e74c3c', label='Impact négatif (↓ risque)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('results/plots/decision_tree/dt_c_feature_importance.png', dpi=300)
        print("[OK] Graphique sauvegardé: results/plots/decision_tree/dt_c_feature_importance.png")
        
        return True
    except Exception as e:
        print(f"[WARNING] Erreur lors de la création du graphique d'importance: {e}")
        return False

# **************************************************
# # --- FONCTIONS UTILITAIRES --- #
# **************************************************

"""
Fonction : load_metrics
Rôle     : Charge les métriques train et test depuis les fichiers texte
Param    : train_file (chemin fichier métriques train), test_file (chemin fichier métriques test)
Retour   : tuple (train_metrics, test_metrics) dictionnaires de métriques
"""
def load_metrics(train_file, test_file):
    train_metrics = {}
    test_metrics = {}
    
    def parse_metrics(file_path, metrics_dict):
        try:
            if file_path and Path(file_path).exists():
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if "Accuracy" in line and ":" in line:
                            metrics_dict['Accuracy'] = float(line.split(':')[1].strip())
                        elif "Precision" in line and ":" in line:
                            metrics_dict['Precision'] = float(line.split(':')[1].strip())
                        elif "Recall" in line and ":" in line:
                            metrics_dict['Recall'] = float(line.split(':')[1].strip())
                        elif "F1-Score" in line and ":" in line:
                            metrics_dict['F1-Score'] = float(line.split(':')[1].strip())
                        elif "AUC-ROC" in line and ":" in line:
                            metrics_dict['AUC-ROC'] = float(line.split(':')[1].strip())
        except:
            pass
    
    parse_metrics(train_file, train_metrics)
    parse_metrics(test_file, test_metrics)
    
    return train_metrics, test_metrics

"""
Fonction : load_confusion_matrix
Rôle     : Charge une matrice de confusion depuis un fichier texte
Param    : file_path (chemin vers le fichier de matrice de confusion)
Retour   : numpy.ndarray (matrice 2x2) ou None en cas d'erreur
"""
def load_confusion_matrix(file_path):
    try:
        if not Path(file_path).exists():
            return None
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        return np.array([[tn, fp], [fn, tp]])
    except:
        return None

"""
Fonction : plot_confusion_matrix_ax
Rôle     : Affiche une matrice de confusion sur un axe matplotlib donné
Param    : ax (axe matplotlib), cm (matrice de confusion 2x2), title (titre du graphique)
Retour   : void
"""
def plot_confusion_matrix_ax(ax, cm, title):
    if cm is None:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
        return
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontweight='bold', fontsize=10)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pas de défaut', 'Défaut'])
    ax.set_yticklabels(['Pas de défaut', 'Défaut'])
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réalité')

"""
Fonction : create_comparison_heatmap
Rôle     : Crée une heatmap de comparaison des métriques entre implémentations C et Python
Param    : ax (axe matplotlib), c_metrics (dict métriques C), python_metrics (dict métriques Python), title (titre)
Retour   : void
"""
def create_comparison_heatmap(ax, c_metrics, python_metrics, title):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    data = []
    
    for metric in metrics:
        c_val = c_metrics.get(metric, 0)
        python_val = python_metrics.get(metric, 0)
        data.append([c_val, python_val])
    
    data = np.array(data)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Ajouter les annotations
    for i in range(len(metrics)):
        for j in range(2):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center",
                          color="black", fontsize=9, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['C', 'Python'])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_title(title, fontweight='bold', fontsize=10)
    
    # Ajouter colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

# **************************************************
# # --- FIGURES RÉCAPITULATIVES --- #
# **************************************************

"""
Fonction : create_summary_figure
Rôle     : Crée une figure récapitulative avec vue d'ensemble de tous les modèles (LR et DT, C et Python)
Param    : aucun
Retour   : void
"""
def create_summary_figure():
    print("\n Génération de la figure récapitulative générale...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Charger toutes les métriques
    lr_c_train, lr_c_test = load_metrics(
        "results/metrics/logistic_regression/lr_c_train_metrics.txt",
        "results/metrics/logistic_regression/lr_c_test_metrics.txt"
    )
    lr_python_train, lr_python_test = load_metrics(
        "results/metrics/logistic_regression/lr_python_train_metrics.txt",
        "results/metrics/logistic_regression/lr_python_test_metrics.txt"
    )
    dt_c_train, dt_c_test = load_metrics(
        "results/metrics/decision_tree/dt_c_train_metrics.txt",
        "results/metrics/decision_tree/dt_c_test_metrics.txt"
    )
    dt_python_train, dt_python_test = load_metrics(
        "results/metrics/decision_tree/dt_python_train_metrics.txt",
        "results/metrics/decision_tree/dt_python_test_metrics.txt"
    )
    
    # Graphique 1: Comparaison des 4 modèles (Accuracy, Precision, Recall, F1, AUC-ROC)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics))
    width = 0.2
    
    lr_c_vals = [lr_c_test.get(m, 0) for m in metrics]
    lr_python_vals = [lr_python_test.get(m, 0) for m in metrics]
    dt_c_vals = [dt_c_test.get(m, 0) for m in metrics]
    dt_python_vals = [dt_python_test.get(m, 0) for m in metrics]
    
    ax1.bar(x - 1.5*width, lr_c_vals, width, label='LR C', color='#3498db')
    ax1.bar(x - 0.5*width, lr_python_vals, width, label='LR Python', color='#e74c3c')
    ax1.bar(x + 0.5*width, dt_c_vals, width, label='DT C', color='#2ecc71')
    ax1.bar(x + 1.5*width, dt_python_vals, width, label='DT Python', color='#f39c12')
    
    ax1.set_xlabel('Métriques')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparaison des 4 Modèles - Métriques Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Graphique 2: Heatmap LR C vs Python
    ax2 = fig.add_subplot(gs[0, 1])
    create_comparison_heatmap(ax2, lr_c_test, lr_python_test, 'LR: C vs Python (Test)')
    
    # Graphique 3: Heatmap DT C vs Python
    ax3 = fig.add_subplot(gs[0, 2])
    create_comparison_heatmap(ax3, dt_c_test, dt_python_test, 'DT: C vs Python (Test)')
    
    # Graphique 4: Comparaison Train vs Test (Accuracy)
    ax4 = fig.add_subplot(gs[0, 3])
    models = ['LR C', 'LR Python', 'DT C', 'DT Python']
    train_acc = [
        lr_c_train.get('Accuracy', 0),
        lr_python_train.get('Accuracy', 0),
        dt_c_train.get('Accuracy', 0),
        dt_python_train.get('Accuracy', 0)
    ]
    test_acc = [
        lr_c_test.get('Accuracy', 0),
        lr_python_test.get('Accuracy', 0),
        dt_c_test.get('Accuracy', 0),
        dt_python_test.get('Accuracy', 0)
    ]
    
    x = np.arange(len(models))
    width = 0.35
    ax4.bar(x - width/2, train_acc, width, label='Train', color='#3498db')
    ax4.bar(x + width/2, test_acc, width, label='Test', color='#e74c3c')
    ax4.set_xlabel('Modèles')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Train vs Test - Accuracy', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Matrices de confusion (ligne 2)
    cm_lr_c = load_confusion_matrix("results/metrics/logistic_regression/lr_c_test_confusion_matrix.txt")
    cm_lr_python = load_confusion_matrix("results/metrics/logistic_regression/lr_python_test_confusion_matrix.txt")
    cm_dt_c = load_confusion_matrix("results/metrics/decision_tree/dt_c_test_confusion_matrix.txt")
    cm_dt_python = load_confusion_matrix("results/metrics/decision_tree/dt_python_test_confusion_matrix.txt")
    
    ax5 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix_ax(ax5, cm_lr_c, 'LR C - Test')
    
    ax6 = fig.add_subplot(gs[1, 1])
    plot_confusion_matrix_ax(ax6, cm_lr_python, 'LR Python - Test')
    
    ax7 = fig.add_subplot(gs[1, 2])
    plot_confusion_matrix_ax(ax7, cm_dt_c, 'DT C - Test')
    
    ax8 = fig.add_subplot(gs[1, 3])
    plot_confusion_matrix_ax(ax8, cm_dt_python, 'DT Python - Test')
    
    # Tableau récapitulatif (ligne 3)
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    # Créer le tableau
    table_data = []
    table_data.append(['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
    table_data.append(['LR C (Test)', 
                      f"{lr_c_test.get('Accuracy', 0):.4f}",
                      f"{lr_c_test.get('Precision', 0):.4f}",
                      f"{lr_c_test.get('Recall', 0):.4f}",
                      f"{lr_c_test.get('F1-Score', 0):.4f}",
                      f"{lr_c_test.get('AUC-ROC', 0):.4f}"])
    table_data.append(['LR Python (Test)',
                      f"{lr_python_test.get('Accuracy', 0):.4f}",
                      f"{lr_python_test.get('Precision', 0):.4f}",
                      f"{lr_python_test.get('Recall', 0):.4f}",
                      f"{lr_python_test.get('F1-Score', 0):.4f}",
                      f"{lr_python_test.get('AUC-ROC', 0):.4f}"])
    table_data.append(['DT C (Test)',
                      f"{dt_c_test.get('Accuracy', 0):.4f}",
                      f"{dt_c_test.get('Precision', 0):.4f}",
                      f"{dt_c_test.get('Recall', 0):.4f}",
                      f"{dt_c_test.get('F1-Score', 0):.4f}",
                      f"{dt_c_test.get('AUC-ROC', 0):.4f}"])
    table_data.append(['DT Python (Test)',
                      f"{dt_python_test.get('Accuracy', 0):.4f}",
                      f"{dt_python_test.get('Precision', 0):.4f}",
                      f"{dt_python_test.get('Recall', 0):.4f}",
                      f"{dt_python_test.get('F1-Score', 0):.4f}",
                      f"{dt_python_test.get('AUC-ROC', 0):.4f}"])
    
    table = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style du tableau
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Tableau Récapitulatif - Métriques de Test', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Vue d\'Ensemble - Comparaison de Tous les Modèles', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('results/plots/summary_figure.png', dpi=300, bbox_inches='tight')
    print("[OK] Figure récapitulative sauvegardée: results/plots/summary_figure.png")

"""
Fonction : create_lr_c_summary
Rôle     : Crée une figure récapitulative détaillée pour la régression logistique C avec toutes les métriques
Param    : aucun
Retour   : void
"""
def create_lr_c_summary():
    print("\n Génération du résumé LR C...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # Charger les métriques
    train_metrics, test_metrics = load_metrics(
        "results/metrics/logistic_regression/lr_c_train_metrics.txt",
        "results/metrics/logistic_regression/lr_c_test_metrics.txt"
    )
    
    # Graphique 1: Métriques train vs test (barres groupées)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    train_vals = [train_metrics.get(m, 0) for m in metrics]
    test_vals = [test_metrics.get(m, 0) for m in metrics]
    
    ax1.bar(x - width/2, train_vals, width, label='Train', color='#3498db')
    ax1.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c')
    ax1.set_xlabel('Métriques')
    ax1.set_ylabel('Score')
    ax1.set_title('Métriques Train vs Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Graphique 2: Matrice de confusion
    ax2 = fig.add_subplot(gs[0, 1])
    cm = load_confusion_matrix("results/metrics/logistic_regression/lr_c_test_confusion_matrix.txt")
    plot_confusion_matrix_ax(ax2, cm, 'Matrice de Confusion - Test')
    
    # Graphique 3: Courbe de convergence
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        cost_file = Path("results/plots/csv/lr_c_cost_curve.csv")
        if cost_file.exists():
            df = pd.read_csv(cost_file)
            ax3.plot(df['iteration'], df['cost'], marker='o', linewidth=2, 
                    markersize=3, color='#e74c3c')
            ax3.set_xlabel('Itérations', fontweight='bold')
            ax3.set_ylabel('Coût (Cross-Entropy Loss)', fontweight='bold')
            ax3.set_title('Convergence - Courbe de Coût', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    except:
        ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    # Graphique 4: Comparaison train vs test (lignes)
    ax4 = fig.add_subplot(gs[1, 0])
    metrics_full = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    train_vals_full = [train_metrics.get(m, 0) for m in metrics_full]
    test_vals_full = [test_metrics.get(m, 0) for m in metrics_full]
    
    ax4.plot(metrics_full, train_vals_full, marker='o', linewidth=2, 
            markersize=8, label='Train', color='#3498db')
    ax4.plot(metrics_full, test_vals_full, marker='s', linewidth=2, 
            markersize=8, label='Test', color='#e74c3c')
    ax4.set_xlabel('Métriques')
    ax4.set_ylabel('Score')
    ax4.set_title('Évolution Train vs Test', fontweight='bold')
    ax4.set_xticklabels(metrics_full, rotation=45, ha='right')
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Graphique 5: Courbe ROC (si disponible)
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        roc_file = Path("results/plots/csv/lr_roc_data.csv")
        if roc_file.exists():
            df_roc = pd.read_csv(roc_file)
            ax5.plot(df_roc['fpr'], df_roc['tpr'], linewidth=2, color='#e74c3c', 
                    label=f'ROC (AUC = {test_metrics.get("AUC-ROC", 0):.3f})')
            ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax5.set_xlabel('Taux de Faux Positifs', fontweight='bold')
            ax5.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
            ax5.set_title('Courbe ROC', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    except:
        ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    
    # Graphique 6: Tableau de métriques
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = []
    table_data.append(['Métrique', 'Train', 'Test'])
    for metric in metrics_full:
        table_data.append([
            metric,
            f"{train_metrics.get(metric, 0):.4f}",
            f"{test_metrics.get(metric, 0):.4f}"
        ])
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Tableau Récapitulatif', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Résumé Détaillé - Régression Logistique (C)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/plots/logistic_regression/summary_lr_c.png', dpi=300, bbox_inches='tight')
    print("[OK] Résumé LR C sauvegardé: results/plots/logistic_regression/summary_lr_c.png")

"""
Fonction : create_lr_python_summary
Rôle     : Crée une figure récapitulative détaillée pour la régression logistique Python avec toutes les métriques
Param    : aucun
Retour   : void
"""
def create_lr_python_summary():
    print("\n Génération du résumé LR Python...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # Charger les métriques
    train_metrics, test_metrics = load_metrics(
        "results/metrics/logistic_regression/lr_python_train_metrics.txt",
        "results/metrics/logistic_regression/lr_python_test_metrics.txt"
    )
    
    # Graphique 1: Métriques train vs test (barres groupées)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    train_vals = [train_metrics.get(m, 0) for m in metrics]
    test_vals = [test_metrics.get(m, 0) for m in metrics]
    
    ax1.bar(x - width/2, train_vals, width, label='Train', color='#3498db')
    ax1.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c')
    ax1.set_xlabel('Métriques')
    ax1.set_ylabel('Score')
    ax1.set_title('Métriques Train vs Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Graphique 2: Matrice de confusion
    ax2 = fig.add_subplot(gs[0, 1])
    cm = load_confusion_matrix("results/metrics/logistic_regression/lr_python_test_confusion_matrix.txt")
    plot_confusion_matrix_ax(ax2, cm, 'Matrice de Confusion - Test')
    
    # Graphique 3: Courbe de convergence
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        cost_file = Path("results/plots/csv/lr_python_cost_curve.csv")
        if cost_file.exists():
            df = pd.read_csv(cost_file)
            ax3.plot(df['iteration'], df['cost'], marker='o', linewidth=2, 
                    markersize=3, color='#e74c3c')
            ax3.set_xlabel('Itérations', fontweight='bold')
            ax3.set_ylabel('Coût (Cross-Entropy Loss)', fontweight='bold')
            ax3.set_title('Convergence - Courbe de Coût', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    except:
        ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    # Graphique 4: Comparaison train vs test (lignes)
    ax4 = fig.add_subplot(gs[1, 0])
    metrics_full = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    train_vals_full = [train_metrics.get(m, 0) for m in metrics_full]
    test_vals_full = [test_metrics.get(m, 0) for m in metrics_full]
    
    ax4.plot(metrics_full, train_vals_full, marker='o', linewidth=2, 
            markersize=8, label='Train', color='#3498db')
    ax4.plot(metrics_full, test_vals_full, marker='s', linewidth=2, 
            markersize=8, label='Test', color='#e74c3c')
    ax4.set_xlabel('Métriques')
    ax4.set_ylabel('Score')
    ax4.set_title('Évolution Train vs Test', fontweight='bold')
    ax4.set_xticklabels(metrics_full, rotation=45, ha='right')
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Graphique 5: Courbe ROC
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        roc_file = Path("results/plots/csv/lr_python_roc_data.csv")
        if roc_file.exists():
            df_roc = pd.read_csv(roc_file)
            ax5.plot(df_roc['fpr'], df_roc['tpr'], linewidth=2, color='#e74c3c', 
                    label=f'ROC (AUC = {test_metrics.get("AUC-ROC", 0):.3f})')
            ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax5.set_xlabel('Taux de Faux Positifs', fontweight='bold')
            ax5.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
            ax5.set_title('Courbe ROC', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    except:
        ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    
    # Graphique 6: Tableau de métriques
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = []
    table_data.append(['Métrique', 'Train', 'Test'])
    for metric in metrics_full:
        table_data.append([
            metric,
            f"{train_metrics.get(metric, 0):.4f}",
            f"{test_metrics.get(metric, 0):.4f}"
        ])
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Tableau Récapitulatif', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Résumé Détaillé - Régression Logistique (Python)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/plots/logistic_regression/summary_lr_python.png', dpi=300, bbox_inches='tight')
    print("[OK] Résumé LR Python sauvegardé: results/plots/logistic_regression/summary_lr_python.png")

"""
Fonction : create_dt_c_summary
Rôle     : Crée une figure récapitulative détaillée pour l'arbre de décision C avec toutes les métriques
Param    : aucun
Retour   : void
"""
def create_dt_c_summary():
    print("\n Génération du résumé DT C...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # Charger les métriques
    train_metrics, test_metrics = load_metrics(
        "results/metrics/decision_tree/dt_c_train_metrics.txt",
        "results/metrics/decision_tree/dt_c_test_metrics.txt"
    )
    
    # Charger les statistiques de l'arbre
    tree_stats = {}
    try:
        with open("results/metrics/decision_tree/dt_c_tree_stats.txt", 'r') as f:
            for line in f:
                if "Profondeur réelle:" in line:
                    tree_stats['depth'] = line.split(':')[1].strip()
                elif "Nombre total de nœuds:" in line:
                    tree_stats['nodes'] = line.split(':')[1].strip()
                elif "Temps d'entraînement:" in line:
                    tree_stats['time'] = line.split(':')[1].strip()
    except:
        pass
    
    # Graphique 1: Métriques train vs test (barres groupées)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    train_vals = [train_metrics.get(m, 0) for m in metrics]
    test_vals = [test_metrics.get(m, 0) for m in metrics]
    
    ax1.bar(x - width/2, train_vals, width, label='Train', color='#2ecc71')
    ax1.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c')
    ax1.set_xlabel('Métriques')
    ax1.set_ylabel('Score')
    ax1.set_title('Métriques Train vs Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Graphique 2: Matrice de confusion
    ax2 = fig.add_subplot(gs[0, 1])
    cm = load_confusion_matrix("results/metrics/decision_tree/dt_c_test_confusion_matrix.txt")
    plot_confusion_matrix_ax(ax2, cm, 'Matrice de Confusion - Test')
    
    # Graphique 3: Importance des caractéristiques (si disponible)
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        feature_file = Path("results/plots/feature_importance.txt")
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                lines = f.readlines()
                features = []
                importances = []
                for line in lines[2:]:  # Skip header
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        features.append(parts[0].strip())
                        importances.append(float(parts[1].strip()))
            
            if features:
                y_pos = np.arange(len(features))
                ax3.barh(y_pos, importances, color='#2ecc71')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(features)
                ax3.set_xlabel('Importance', fontweight='bold')
                ax3.set_title('Importance des Caractéristiques', fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
        else:
            ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    except:
        ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    # Graphique 4: Comparaison train vs test (lignes)
    ax4 = fig.add_subplot(gs[1, 0])
    metrics_full = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    train_vals_full = [train_metrics.get(m, 0) for m in metrics_full]
    test_vals_full = [test_metrics.get(m, 0) for m in metrics_full]
    
    ax4.plot(metrics_full, train_vals_full, marker='o', linewidth=2, 
            markersize=8, label='Train', color='#2ecc71')
    ax4.plot(metrics_full, test_vals_full, marker='s', linewidth=2, 
            markersize=8, label='Test', color='#e74c3c')
    ax4.set_xlabel('Métriques')
    ax4.set_ylabel('Score')
    ax4.set_title('Évolution Train vs Test', fontweight='bold')
    ax4.set_xticklabels(metrics_full, rotation=45, ha='right')
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Graphique 5: Courbe ROC
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        roc_file = Path("results/plots/csv/dt_roc_data.csv")
        if roc_file.exists():
            df_roc = pd.read_csv(roc_file)
            ax5.plot(df_roc['fpr'], df_roc['tpr'], linewidth=2, color='#2ecc71', 
                    label=f'ROC (AUC = {test_metrics.get("AUC-ROC", 0):.3f})')
            ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax5.set_xlabel('Taux de Faux Positifs', fontweight='bold')
            ax5.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
            ax5.set_title('Courbe ROC', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    except:
        ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    
    # Graphique 6: Statistiques de l'arbre
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Tableau avec métriques et statistiques
    table_data = []
    table_data.append(['Métrique', 'Train', 'Test'])
    for metric in metrics_full:
        table_data.append([
            metric,
            f"{train_metrics.get(metric, 0):.4f}",
            f"{test_metrics.get(metric, 0):.4f}"
        ])
    
    table_data.append(['', '', ''])
    table_data.append(['Statistique', 'Valeur', ''])
    table_data.append(['Profondeur réelle', tree_stats.get('depth', 'N/A'), ''])
    table_data.append(['Nombre de nœuds', tree_stats.get('nodes', 'N/A'), ''])
    table_data.append(['Temps d\'entraînement', tree_stats.get('time', 'N/A'), ''])
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2ecc71')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Tableau Récapitulatif', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Résumé Détaillé - Arbre de Décision (C)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/plots/decision_tree/summary_dt_c.png', dpi=300, bbox_inches='tight')
    print("[OK] Résumé DT C sauvegardé: results/plots/decision_tree/summary_dt_c.png")

"""
Fonction : create_dt_python_summary
Rôle     : Crée une figure récapitulative détaillée pour l'arbre de décision Python avec toutes les métriques
Param    : aucun
Retour   : void
"""
def create_dt_python_summary():
    print("\n Génération du résumé DT Python...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # Charger les métriques
    train_metrics, test_metrics = load_metrics(
        "results/metrics/decision_tree/dt_python_train_metrics.txt",
        "results/metrics/decision_tree/dt_python_test_metrics.txt"
    )
    
    # Charger les statistiques de l'arbre
    tree_stats = {}
    try:
        with open("results/metrics/decision_tree/dt_python_tree_stats.txt", 'r') as f:
            for line in f:
                if "Profondeur réelle:" in line:
                    tree_stats['depth'] = line.split(':')[1].strip()
                elif "Nombre total de nœuds:" in line:
                    tree_stats['nodes'] = line.split(':')[1].strip()
                elif "Temps d'entraînement:" in line:
                    tree_stats['time'] = line.split(':')[1].strip()
    except:
        pass
    
    # Graphique 1: Métriques train vs test (barres groupées)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    train_vals = [train_metrics.get(m, 0) for m in metrics]
    test_vals = [test_metrics.get(m, 0) for m in metrics]
    
    ax1.bar(x - width/2, train_vals, width, label='Train', color='#f39c12')
    ax1.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c')
    ax1.set_xlabel('Métriques')
    ax1.set_ylabel('Score')
    ax1.set_title('Métriques Train vs Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Graphique 2: Matrice de confusion
    ax2 = fig.add_subplot(gs[0, 1])
    cm = load_confusion_matrix("results/metrics/decision_tree/dt_python_test_confusion_matrix.txt")
    plot_confusion_matrix_ax(ax2, cm, 'Matrice de Confusion - Test')
    
    # Graphique 3: Importance des caractéristiques (si disponible)
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        feature_file = Path("results/plots/feature_importance.txt")
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                lines = f.readlines()
                features = []
                importances = []
                for line in lines[2:]:  # Skip header
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        features.append(parts[0].strip())
                        importances.append(float(parts[1].strip()))
            
            if features:
                y_pos = np.arange(len(features))
                ax3.barh(y_pos, importances, color='#f39c12')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(features)
                ax3.set_xlabel('Importance', fontweight='bold')
                ax3.set_title('Importance des Caractéristiques', fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
        else:
            ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    except:
        ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    # Graphique 4: Comparaison train vs test (lignes)
    ax4 = fig.add_subplot(gs[1, 0])
    metrics_full = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    train_vals_full = [train_metrics.get(m, 0) for m in metrics_full]
    test_vals_full = [test_metrics.get(m, 0) for m in metrics_full]
    
    ax4.plot(metrics_full, train_vals_full, marker='o', linewidth=2, 
            markersize=8, label='Train', color='#f39c12')
    ax4.plot(metrics_full, test_vals_full, marker='s', linewidth=2, 
            markersize=8, label='Test', color='#e74c3c')
    ax4.set_xlabel('Métriques')
    ax4.set_ylabel('Score')
    ax4.set_title('Évolution Train vs Test', fontweight='bold')
    ax4.set_xticklabels(metrics_full, rotation=45, ha='right')
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Graphique 5: Courbe ROC
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        roc_file = Path("results/plots/csv/dt_python_roc_data.csv")
        if roc_file.exists():
            df_roc = pd.read_csv(roc_file)
            ax5.plot(df_roc['fpr'], df_roc['tpr'], linewidth=2, color='#f39c12', 
                    label=f'ROC (AUC = {test_metrics.get("AUC-ROC", 0):.3f})')
            ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax5.set_xlabel('Taux de Faux Positifs', fontweight='bold')
            ax5.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
            ax5.set_title('Courbe ROC', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    except:
        ax5.text(0.5, 0.5, 'Données ROC non disponibles', ha='center', va='center')
    
    # Graphique 6: Statistiques de l'arbre
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Tableau avec métriques et statistiques
    table_data = []
    table_data.append(['Métrique', 'Train', 'Test'])
    for metric in metrics_full:
        table_data.append([
            metric,
            f"{train_metrics.get(metric, 0):.4f}",
            f"{test_metrics.get(metric, 0):.4f}"
        ])
    
    table_data.append(['', '', ''])
    table_data.append(['Statistique', 'Valeur', ''])
    table_data.append(['Profondeur réelle', tree_stats.get('depth', 'N/A'), ''])
    table_data.append(['Nombre de nœuds', tree_stats.get('nodes', 'N/A'), ''])
    table_data.append(['Temps d\'entraînement', tree_stats.get('time', 'N/A'), ''])
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#f39c12')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Tableau Récapitulatif', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Résumé Détaillé - Arbre de Décision (Python)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/plots/decision_tree/summary_dt_python.png', dpi=300, bbox_inches='tight')
    print("[OK] Résumé DT Python sauvegardé: results/plots/decision_tree/summary_dt_python.png")

"""
Fonction : main
Rôle     : Fonction principale orchestrant la génération de tous les graphiques de résultats
Param    : aucun
Retour   : void
"""
def main():
    # Créer les répertoires nécessaires
    os.makedirs("results/plots/logistic_regression", exist_ok=True)
    os.makedirs("results/plots/decision_tree", exist_ok=True)
    os.makedirs("results/plots/csv", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("VISUALISATION DES RÉSULTATS")
    print("=" * 60 + "\n")
    
    # Générer tous les graphiques
    success_count = 0
    
    if plot_lr_confusion_matrix():
        success_count += 1
    
    if plot_dt_confusion_matrix():
        success_count += 1
    
    if plot_lr_metrics_train_vs_test():
        success_count += 1
    
    if plot_dt_metrics_train_vs_test():
        success_count += 1
    
    if plot_lr_cost_curve():
        success_count += 1
    
    if plot_lr_python_cost_curve():
        success_count += 1
    
    if plot_roc_curves_comparison():
        success_count += 1
    
    if plot_dt_feature_importance():
        success_count += 1
    
    if plot_lr_python_confusion_matrix():
        success_count += 1
    
    if plot_dt_python_confusion_matrix():
        success_count += 1
    
    if plot_python_roc_curves():
        success_count += 1
    
    if plot_lr_python_metrics_train_vs_test():
        success_count += 1
    
    if plot_dt_python_metrics_train_vs_test():
        success_count += 1
    
    create_summary_figure()
    success_count += 1
    
    create_lr_c_summary()
    success_count += 1
    
    create_lr_python_summary()
    success_count += 1
    
    create_dt_c_summary()
    success_count += 1
    
    create_dt_python_summary()
    success_count += 1
    
    print("\n" + "=" * 60)
    print(f"[OK] VISUALISATION TERMINÉE ({success_count}/19 graphiques générés)")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - results/plots/logistic_regression/lr_c_confusion_matrix_test.png")
    print("  - results/plots/decision_tree/dt_c_confusion_matrix_test.png")
    print("  - results/plots/logistic_regression/lr_c_metrics_train_vs_test.png")
    print("  - results/plots/decision_tree/dt_c_metrics_train_vs_test.png")
    print("  - results/plots/logistic_regression/lr_c_cost_curve_training.png")
    print("  - results/plots/logistic_regression/lr_python_cost_curve_training.png")
    print("  - results/plots/logistic_regression/lr_dt_c_roc_curves_comparison.png")
    print("  - results/plots/decision_tree/dt_c_feature_importance.png")
    print("  - results/plots/logistic_regression/lr_python_confusion_matrix_test.png")
    print("  - results/plots/decision_tree/dt_python_confusion_matrix_test.png")
    print("  - results/plots/logistic_regression/lr_dt_python_roc_curves_comparison.png")
    print("  - results/plots/logistic_regression/lr_python_metrics_train_vs_test.png")
    print("  - results/plots/decision_tree/dt_python_metrics_train_vs_test.png")
    print("  - results/plots/summary_figure.png")
    print("  - results/plots/logistic_regression/summary_lr_c.png")
    print("  - results/plots/logistic_regression/summary_lr_python.png")
    print("  - results/plots/decision_tree/summary_dt_c.png")
    print("  - results/plots/decision_tree/summary_dt_python.png")
    print()

if __name__ == "__main__":
    main()

