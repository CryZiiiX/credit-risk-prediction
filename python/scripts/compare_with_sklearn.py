#!/usr/bin/env python3
"""
/*****************************************************************************************************

Nom : scripts/compare_with_sklearn.py

Rôle : Script de comparaison de l'implémentation C avec scikit-learn

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : N/A (script Python)

    Pour executer : python3 scripts/compare_with_sklearn.py

******************************************************************************************************/
"""

import pandas as pd
import numpy as np
import time
import os
# train_test_split n'est plus utilisé (split fait uniquement en C)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from pathlib import Path

# **************************************************
# # --- CHARGEMENT ET PRÉTRAITEMENT --- #
# **************************************************

"""
Fonction : load_train_test_from_files
Rôle     : Charge les datasets train.csv et test.csv générés par le programme C
Param    : aucun
Retour   : tuple (X_train, X_test, y_train, y_test) ou None si fichiers inexistants
"""
def load_train_test_from_files():
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")
    
    if not train_path.exists() or not test_path.exists():
        return None
    
    print(" Chargement des datasets train.csv et test.csv générés par C...")
    
    # Charger les fichiers CSV
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"[OK] Train: {len(df_train)} échantillons, Test: {len(df_test)} échantillons")
    
    # Séparer X et y (loan_status est à l'index 8 dans le header)
    # Header: person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length
    X_train = df_train.drop('loan_status', axis=1)
    y_train = df_train['loan_status']
    X_test = df_test.drop('loan_status', axis=1)
    y_test = df_test['loan_status']
    
    print("[OK] Datasets chargés et séparés (X, y)")
    
    return X_train, X_test, y_train, y_test

"""
Fonction : load_and_preprocess_data
Rôle     : Charge le dataset CSV et applique le prétraitement identique à l'implémentation C
Param    : aucun
Retour   : tuple (X, y) features et labels prétraités
"""
def load_and_preprocess_data():
    print(" Chargement du dataset...")
    df = pd.read_csv("data/raw/credit_risk_dataset.csv")
    
    print(f"[OK] Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Encodage des variables catégorielles (comme dans encoder.c)
    print("\n Encodage des variables catégorielles...")
    
    # person_home_ownership: RENT=0, OWN=1, MORTGAGE=2, OTHER=3
    home_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    df['person_home_ownership'] = df['person_home_ownership'].map(home_mapping)
    
    # loan_intent: PERSONAL=0, EDUCATION=1, MEDICAL=2, VENTURE=3, HOMEIMPROVEMENT=4, DEBTCONSOLIDATION=5
    intent_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 
                     'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    df['loan_intent'] = df['loan_intent'].map(intent_mapping)
    
    # loan_grade: A=1, B=2, C=3, D=4, E=5, F=6, G=7
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['loan_grade'] = df['loan_grade'].map(grade_mapping)
    
    # cb_person_default_on_file: N=0, Y=1
    default_mapping = {'N': 0, 'Y': 1}
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_mapping)
    
    print("[OK] Variables catégorielles encodées")
    
    # Gestion des valeurs manquantes (remplacement par la moyenne)
    print("\n Gestion des valeurs manquantes...")
    df = df.fillna(df.mean())
    print("[OK] Valeurs manquantes traitées")
    
    # Séparer X et y
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    return X, y

"""
Fonction : train_sklearn_model
Rôle     : Entraîne un modèle de régression logistique scikit-learn et suit la courbe de coût
Param    : X_train (features train), y_train (labels train), X_test (features test), y_test (labels test)
Retour   : tuple (model, scaler, X_train_scaled, X_test_scaled)
"""
def train_sklearn_model(X_train, y_train, X_test, y_test):
    print("\n Entraînement du modèle scikit-learn...")
    
    # Normalisation (StandardScaler comme dans scaler.c)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Créer le répertoire pour sauvegarder la courbe de coût
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # Points d'itération où calculer la loss (similaire à C qui sauvegarde tous les 100)
    # Note: on commence à 1 au lieu de 0 car max_iter=0 cause des problèmes de convergence
    iteration_points = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    cost_data = []
    
    # Entraîner à différents points d'itération pour suivre la loss
    for max_iter in iteration_points:
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Calculer la loss sur le train set
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        loss = calculate_cross_entropy_loss(y_train, y_train_proba)
        cost_data.append({'iteration': max_iter, 'cost': loss})
    
    # Utiliser le modèle final (max_iter=1000)
    final_model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    final_model.fit(X_train_scaled, y_train)
    
    # Sauvegarder la courbe de coût
    os.makedirs("results/plots/csv", exist_ok=True)
    cost_df = pd.DataFrame(cost_data)
    cost_df.to_csv("results/plots/csv/lr_python_cost_curve.csv", index=False)
    print("[OK] Courbe de coût sauvegardée: results/plots/csv/lr_python_cost_curve.csv")
    
    print("[OK] Modèle entraîné")
    
    return final_model, scaler, X_train_scaled, X_test_scaled

"""
Fonction : train_sklearn_decision_tree
Rôle     : Entraîne un arbre de décision scikit-learn avec les mêmes hyperparamètres que l'implémentation C
Param    : X_train (features train), y_train (labels train), X_test (features test), y_test (labels test)
Retour   : tuple (model, training_time) modèle entraîné et temps d'entraînement
"""
def train_sklearn_decision_tree(X_train, y_train, X_test, y_test):
    print("\n Entraînement de l'arbre de décision scikit-learn...")
    
    # Note : L'arbre de décision C n'utilise pas de normalisation
    # Donc on utilise les données brutes (non normalisées)
    
    # Arbre de décision avec paramètres correspondant à l'implémentation C:
    # - max_depth=7 (comme dans main.c ligne 145)
    # - min_samples_split=20 (comme dans main.c ligne 145)
    # - min_samples_leaf=10 (comme dans main.c ligne 145)
    # - criterion='gini' (comme GINI dans main.c ligne 145)
    # - random_state=42 (pour reproductibilité)
    model = DecisionTreeClassifier(
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='gini',
        random_state=42
    )
    
    # Mesurer le temps d'entraînement
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    print("[OK] Arbre de décision entraîné")
    
    return model, training_time

# **************************************************
# # --- CHARGEMENT DES RÉSULTATS --- #
# **************************************************

"""
Fonction : load_c_results
Rôle     : Charge les métriques de test de l'implémentation C depuis les fichiers texte
Param    : aucun
Retour   : dict (métriques C) ou None en cas d'erreur
"""
def load_c_results():
    print("\n Chargement des résultats de l'implémentation C...")
    
    results = {}
    
    # Charger les métriques de test
    try:
        with open("results/metrics/logistic_regression/lr_c_test_metrics.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Accuracy" in line:
                    results['c_accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    results['c_precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    results['c_recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    results['c_f1'] = float(line.split(':')[1].strip())
                elif "AUC-ROC" in line:
                    results['c_auc_roc'] = float(line.split(':')[1].strip())
        
        print("[OK] Métriques C chargées")
    except FileNotFoundError:
        print("[WARNING] Fichiers de résultats C non trouvés. Exécutez d'abord le programme C.")
        return None
    
    return results

"""
Fonction : load_c_decision_tree_results
Rôle     : Charge les métriques de test de l'arbre de décision C depuis les fichiers texte
Param    : aucun
Retour   : dict (métriques C) ou None en cas d'erreur
"""
def load_c_decision_tree_results():
    print("\n Chargement des résultats de l'arbre de décision C...")
    
    results = {}
    
    # Charger les métriques de test de l'arbre de décision
    try:
        with open("results/metrics/decision_tree/dt_c_test_metrics.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Accuracy" in line:
                    results['c_accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    results['c_precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    results['c_recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    results['c_f1'] = float(line.split(':')[1].strip())
                elif "AUC-ROC" in line:
                    results['c_auc_roc'] = float(line.split(':')[1].strip())
        
        print("[OK] Métriques arbre de décision C chargées")
    except FileNotFoundError:
        print("[WARNING] Fichiers de résultats arbre de décision C non trouvés. Exécutez d'abord le programme C.")
        return None
    
    return results

# **************************************************
# # --- COMPARAISON DES RÉSULTATS --- #
# **************************************************

"""
Fonction : compare_results
Rôle     : Compare et affiche les métriques entre scikit-learn et l'implémentation C pour la régression logistique
Param    : sklearn_metrics (dict métriques sklearn), c_results (dict métriques C)
Retour   : void
"""
def compare_results(sklearn_metrics, c_results):
    print("\n" + "=" * 60)
    print("COMPARAISON DES RÉSULTATS")
    print("=" * 60)
    
    if c_results is None:
        print("\n[WARNING] Impossible de comparer : résultats C manquants")
        return
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Métrique", "Scikit-learn", "C (custom)", "Différence"))
    print("-" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for metric, name in zip(metrics, metric_names):
        sklearn_val = sklearn_metrics[metric]
        c_val = c_results[f'c_{metric}']
        diff = abs(sklearn_val - c_val)
        
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(name, sklearn_val, c_val, diff))
    
    print("\n" + "=" * 60)
    
    # Analyse des différences
    print("\n ANALYSE:")
    
    max_diff = max([abs(sklearn_metrics[m] - c_results[f'c_{m}']) for m in metrics])
    
    if max_diff < 0.01:
        print("[OK] Excellent ! Les résultats sont quasiment identiques (différence < 1%)")
    elif max_diff < 0.05:
        print("[OK] Bon ! Les résultats sont très similaires (différence < 5%)")
    elif max_diff < 0.10:
        print("[WARNING] Les résultats sont similaires mais avec quelques différences (< 10%)")
    else:
        print("[WARNING] Les résultats diffèrent significativement (> 10%)")
    
    print("\nCauses possibles de différences:")
    print("  - Différences d'optimisation (L-BFGS vs Gradient Descent)")
    print("  - Précision numérique (float vs double)")
    print("  - Initialisation aléatoire du split train/test")
    print("  - Nombre d'itérations différent avant convergence")

"""
Fonction : compare_decision_trees
Rôle     : Compare et affiche les métriques entre scikit-learn et l'implémentation C pour l'arbre de décision
Param    : sklearn_metrics (dict métriques sklearn), c_results (dict métriques C)
Retour   : void
"""
def compare_decision_trees(sklearn_metrics, c_results):
    print("\n" + "=" * 60)
    print("COMPARAISON ARBRE DE DÉCISION")
    print("=" * 60)
    
    if c_results is None:
        print("\n[WARNING] Impossible de comparer : résultats C manquants")
        return
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Métrique", "Scikit-learn", "C (custom)", "Différence"))
    print("-" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    for metric, name in zip(metrics, metric_names):
        sklearn_val = sklearn_metrics[metric]
        c_val = c_results[f'c_{metric}']
        diff = abs(sklearn_val - c_val)
        
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(name, sklearn_val, c_val, diff))
    
    print("\n" + "=" * 60)
    
    # Analyse des différences
    print("\n ANALYSE:")
    
    max_diff = max([abs(sklearn_metrics[m] - c_results[f'c_{m}']) for m in metrics])
    
    if max_diff < 0.01:
        print("[OK] Excellent ! Les résultats sont quasiment identiques (différence < 1%)")
    elif max_diff < 0.05:
        print("[OK] Bon ! Les résultats sont très similaires (différence < 5%)")
    elif max_diff < 0.10:
        print("[WARNING] Les résultats sont similaires mais avec quelques différences (< 10%)")
    else:
        print("[WARNING] Les résultats diffèrent significativement (> 10%)")
    
    print("\nCauses possibles de différences:")
    print("  - Différences dans l'algorithme de recherche du meilleur split")
    print("  - Précision numérique (float vs double)")
    print("  - Initialisation aléatoire du split train/test")
    print("  - Gestion des cas limites (valeurs manquantes, égalités)")

# **************************************************
# # --- SAUVEGARDE DES RÉSULTATS --- #
# **************************************************

"""
Fonction : save_sklearn_lr_metrics
Rôle     : Sauvegarde les métriques de test de la régression logistique scikit-learn dans un fichier texte
Param    : sklearn_metrics (dict contenant les métriques)
Retour   : void
"""
def save_sklearn_lr_metrics(sklearn_metrics):
    output_path = Path("results/metrics/logistic_regression/lr_python_test_metrics.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REGRESSION LOGISTIQUE - SCIKIT-LEARN (PYTHON)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy:  {sklearn_metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {sklearn_metrics['precision']:.6f}\n")
        f.write(f"Recall:    {sklearn_metrics['recall']:.6f}\n")
        f.write(f"F1-Score:  {sklearn_metrics['f1']:.6f}\n")
        f.write(f"AUC-ROC:   {sklearn_metrics['auc_roc']:.6f}\n")
    
    print(f"[OK] Métriques sklearn LR sauvegardées: {output_path}")

"""
Fonction : save_sklearn_dt_metrics
Rôle     : Sauvegarde les métriques de test de l'arbre de décision scikit-learn dans un fichier texte
Param    : sklearn_metrics (dict contenant les métriques)
Retour   : void
"""
def save_sklearn_dt_metrics(sklearn_metrics):
    output_path = Path("results/metrics/decision_tree/dt_python_test_metrics.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ARBRE DE DÉCISION - SCIKIT-LEARN (PYTHON)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy:  {sklearn_metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {sklearn_metrics['precision']:.6f}\n")
        f.write(f"Recall:    {sklearn_metrics['recall']:.6f}\n")
        f.write(f"F1-Score:  {sklearn_metrics['f1']:.6f}\n")
        f.write(f"AUC-ROC:   {sklearn_metrics['auc_roc']:.6f}\n")
    
    print(f"[OK] Métriques sklearn DT sauvegardées: {output_path}")

"""
Fonction : save_decision_tree_stats
Rôle     : Sauvegarde les statistiques de l'arbre de décision (profondeur, nœuds, temps) dans un fichier
Param    : model (modèle DecisionTreeClassifier), training_time (temps d'entraînement), filename (chemin fichier)
Retour   : void
"""
def save_decision_tree_stats(model, training_time, filename):
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    actual_depth = model.tree_.max_depth
    total_nodes = model.tree_.node_count
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STATISTIQUES ARBRE DE DÉCISION - SCIKIT-LEARN (PYTHON)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Profondeur réelle: {actual_depth}\n")
        f.write(f"Nombre total de nœuds: {total_nodes}\n")
        f.write(f"Temps d'entraînement: environ {training_time:.3f} secondes\n")
        f.write("\n")
        f.write("Hyperparamètres:\n")
        f.write(f"  - max_depth: {model.max_depth}\n")
        f.write(f"  - min_samples_split: {model.min_samples_split}\n")
        f.write(f"  - min_samples_leaf: {model.min_samples_leaf}\n")
        f.write(f"  - criterion: {model.criterion}\n")
    
    print(f"[OK] Statistiques arbre de décision sauvegardées: {output_path}")

"""
Fonction : save_python_train_metrics
Rôle     : Calcule et sauvegarde les métriques d'entraînement dans un fichier texte
Param    : y_true (labels réels), y_pred (labels prédits), filename (chemin fichier)
Retour   : void
"""
def save_python_train_metrics(y_true, y_pred, filename):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1-Score: {f1:.6f}\n")
    print(f"[OK] Métriques train sauvegardées: {output_path}")

"""
Fonction : calculate_cross_entropy_loss
Rôle     : Calcule la perte d'entropie croisée (log loss) pour des probabilités
Param    : y_true (labels réels), y_pred_proba (probabilités prédites)
Retour   : double (valeur de la perte)
"""
def calculate_cross_entropy_loss(y_true, y_pred_proba):
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    return loss

"""
Fonction : save_python_confusion_matrix
Rôle     : Sauvegarde une matrice de confusion au format texte (TN, FP, FN, TP)
Param    : cm (matrice de confusion numpy), filename (chemin fichier)
Retour   : void
"""
def save_python_confusion_matrix(cm, filename):
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"TN: {cm[0,0]}, FP: {cm[0,1]}\n")
        f.write(f"FN: {cm[1,0]}, TP: {cm[1,1]}\n")
    print(f"[OK] Matrice de confusion sauvegardée: {output_path}")

"""
Fonction : save_roc_data
Rôle     : Calcule et sauvegarde les données de la courbe ROC (fpr, tpr) dans un fichier CSV
Param    : y_true (labels réels), y_proba (probabilités prédites), filename (chemin fichier CSV)
Retour   : void
"""
def save_roc_data(y_true, y_proba, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Données ROC sauvegardées: {filename}")

"""
Fonction : save_comparison_report
Rôle     : Sauvegarde un rapport de comparaison détaillé entre C et Python pour la régression logistique
Param    : sklearn_metrics (dict métriques sklearn), c_results (dict métriques C)
Retour   : void
"""
def save_comparison_report(sklearn_metrics, c_results):
    output_path = Path("results/metrics/logistic_regression/lr_comparison_c_vs_python.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("COMPARAISON REGRESSION LOGISTIQUE : C vs PYTHON\n")
        f.write("=" * 60 + "\n\n")
        
        if c_results:
            f.write("{:<20} {:<20} {:<20} {:<15}\n".format("Métrique", "C (custom)", "Python (sklearn)", "Différence"))
            f.write("-" * 75 + "\n")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
            
            for metric, name in zip(metrics, metric_names):
                sklearn_val = sklearn_metrics[metric]
                c_val = c_results[f'c_{metric}']
                diff = abs(sklearn_val - c_val)
                
                f.write("{:<20} {:<20.6f} {:<20.6f} {:<15.6f}\n".format(name, c_val, sklearn_val, diff))
    
    print(f"[OK] Rapport de comparaison sauvegardé: {output_path}")

"""
Fonction : save_decision_tree_comparison_report
Rôle     : Sauvegarde un rapport de comparaison détaillé entre C et Python pour l'arbre de décision
Param    : sklearn_metrics (dict métriques sklearn), c_results (dict métriques C)
Retour   : void
"""
def save_decision_tree_comparison_report(sklearn_metrics, c_results):
    output_path = Path("results/metrics/decision_tree/dt_comparison_c_vs_python.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("COMPARAISON ARBRE DE DÉCISION : C vs PYTHON\n")
        f.write("=" * 60 + "\n\n")
        
        if c_results:
            f.write("{:<20} {:<20} {:<20} {:<15}\n".format("Métrique", "C (custom)", "Python (sklearn)", "Différence"))
            f.write("-" * 75 + "\n")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
            
            for metric, name in zip(metrics, metric_names):
                sklearn_val = sklearn_metrics[metric]
                c_val = c_results[f'c_{metric}']
                diff = abs(sklearn_val - c_val)
                
                f.write("{:<20} {:<20.6f} {:<20.6f} {:<15.6f}\n".format(name, c_val, sklearn_val, diff))
    
    print(f"[OK] Rapport de comparaison arbre de décision sauvegardé: {output_path}")

"""
Fonction : main
Rôle     : Fonction principale orchestrant la comparaison complète entre scikit-learn et l'implémentation C
Param    : aucun
Retour   : void
"""
def main():
    print("\n" + "=" * 60)
    print("VALIDATION AVEC SCIKIT-LEARN")
    print("=" * 60 + "\n")
    
    # Charger train.csv et test.csv générés par C
    result = load_train_test_from_files()
    
    if result is None:
        print("\n[ERREUR] Les fichiers train.csv et test.csv n'existent pas.")
        print("         Veuillez d'abord exécuter le programme C pour générer ces fichiers :")
        print("         ./build/credit_risk_predictor")
        print("         ou")
        print("         make run")
        return
    
    X_train, X_test, y_train, y_test = result
    
    # Entraîner le modèle sklearn
    model, scaler, X_train_scaled, X_test_scaled = train_sklearn_model(
        X_train, y_train, X_test, y_test
    )
    
    # Prédictions sur le train set
    y_train_pred = model.predict(X_train_scaled)
    save_python_train_metrics(y_train, y_train_pred, "results/metrics/logistic_regression/lr_python_train_metrics.txt")
    
    # Prédictions
    print("\n Prédictions sur le test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Sauvegarder les données ROC
    os.makedirs("results/plots/csv", exist_ok=True)
    save_roc_data(y_test, y_pred_proba, "results/plots/csv/lr_python_roc_data.csv")
    
    # Calculer les métriques
    sklearn_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n Métriques scikit-learn:")
    print(f"  Accuracy:  {sklearn_metrics['accuracy']:.4f}")
    print(f"  Precision: {sklearn_metrics['precision']:.4f}")
    print(f"  Recall:    {sklearn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {sklearn_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {sklearn_metrics['auc_roc']:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Sauvegarder la matrice de confusion
    save_python_confusion_matrix(cm, "results/metrics/logistic_regression/lr_python_test_confusion_matrix.txt")
    
    # Sauvegarder les métriques sklearn
    save_sklearn_lr_metrics(sklearn_metrics)
    
    # Charger et comparer avec les résultats C
    c_results = load_c_results()
    compare_results(sklearn_metrics, c_results)
    save_comparison_report(sklearn_metrics, c_results)
    
    # ============================================================
    # COMPARAISON ARBRE DE DÉCISION
    # ============================================================
    
    print("\n\n" + "=" * 60)
    print("COMPARAISON ARBRE DE DÉCISION")
    print("=" * 60 + "\n")
    
    # Entraîner l'arbre de décision sklearn
    dt_model, dt_training_time = train_sklearn_decision_tree(X_train, y_train, X_test, y_test)
    
    # Sauvegarder les statistiques de l'arbre
    save_decision_tree_stats(dt_model, dt_training_time, "results/metrics/decision_tree/dt_python_tree_stats.txt")
    
    # Prédictions sur le train set
    dt_y_train_pred = dt_model.predict(X_train)
    save_python_train_metrics(y_train, dt_y_train_pred, "results/metrics/decision_tree/dt_python_train_metrics.txt")
    
    # Prédictions
    print("\n Prédictions sur le test set...")
    dt_y_pred = dt_model.predict(X_test)
    dt_y_proba = dt_model.predict_proba(X_test)[:, 1]  # Probabilités de classe positive
    
    # Sauvegarder les données ROC
    os.makedirs("results/plots/csv", exist_ok=True)
    save_roc_data(y_test, dt_y_proba, "results/plots/csv/dt_python_roc_data.csv")
    
    # Calculer les métriques
    dt_sklearn_metrics = {
        'accuracy': accuracy_score(y_test, dt_y_pred),
        'precision': precision_score(y_test, dt_y_pred, zero_division=0),
        'recall': recall_score(y_test, dt_y_pred, zero_division=0),
        'f1': f1_score(y_test, dt_y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, dt_y_proba)
    }
    
    print("\n Métriques arbre de décision scikit-learn:")
    print(f"  Accuracy:  {dt_sklearn_metrics['accuracy']:.4f}")
    print(f"  Precision: {dt_sklearn_metrics['precision']:.4f}")
    print(f"  Recall:    {dt_sklearn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {dt_sklearn_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {dt_sklearn_metrics['auc_roc']:.4f}")
    
    # Matrice de confusion
    dt_cm = confusion_matrix(y_test, dt_y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {dt_cm[0,0]}, FP: {dt_cm[0,1]}")
    print(f"    FN: {dt_cm[1,0]}, TP: {dt_cm[1,1]}")
    
    # Sauvegarder la matrice de confusion
    save_python_confusion_matrix(dt_cm, "results/metrics/decision_tree/dt_python_test_confusion_matrix.txt")
    
    # Sauvegarder les métriques sklearn
    save_sklearn_dt_metrics(dt_sklearn_metrics)
    
    # Charger et comparer avec les résultats C de l'arbre de décision
    dt_c_results = load_c_decision_tree_results()
    compare_decision_trees(dt_sklearn_metrics, dt_c_results)
    save_decision_tree_comparison_report(dt_sklearn_metrics, dt_c_results)
    
    print("\n" + "=" * 60)
    print("[OK] COMPARAISON TERMINÉE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

