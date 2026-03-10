# Prédiction du Risque de Crédit Bancaire en C
![Présentation du projet](docs/rapport/PagedeprésentationderapportCANVA.png)

**Projet académique M1 Informatique - Techniques d'Apprentissage Artificiel**

Implémentation from scratch de deux algorithmes de machine learning en C pour la prédiction du risque de défaut de paiement bancaire. Ce projet démontre la mise en œuvre complète d'un pipeline d'apprentissage automatique, depuis le chargement des données jusqu'à l'évaluation des modèles, sans utilisation de bibliothèques d'apprentissage automatique.

## Table des Matières

- [Description](#description)
- [Architecture du Projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats et Performances](#résultats-et-performances)
- [Structure des Fichiers Générés](#structure-des-fichiers-générés)
- [Documentation](#documentation)
- [Auteur](#auteur)

---

## Description

Ce projet académique propose une implémentation complète et autonome d'un système d'apprentissage automatique en langage C. L'objectif principal est de résoudre un problème de classification supervisée binaire pour la prédiction du risque de crédit bancaire.

Le système classifie les emprunteurs en deux catégories :
- **Classe 0** : Absence de défaut de paiement (bon risque)
- **Classe 1** : Présence de défaut de paiement (mauvais risque)

### Contributions Techniques

Le projet se distingue par plusieurs contributions techniques et pédagogiques :

- **Implémentation from scratch** : Deux modèles de classification supervisée (Régression Logistique et Arbre de Décision CART) développés sans bibliothèques d'apprentissage automatique
- **Pipeline complet** : Système intégré de chargement, prétraitement, entraînement et évaluation des données
- **Gestion des données catégorielles** : Encodage automatique de quatre variables catégorielles selon un schéma prédéfini
- **Normalisation des données** : Implémentation du StandardScaler pour la standardisation des variables numériques
- **Division des données** : Split train/test avec ratio 80/20 effectué uniquement par le programme C. Les fichiers `train.csv` et `test.csv` sont sauvegardés et réutilisés par les scripts Python pour garantir l'utilisation des mêmes données
- **Métriques d'évaluation complètes** : Calcul de l'accuracy, précision, rappel, F1-score, AUC-ROC et génération de matrices de confusion
- **Validation expérimentale** : Comparaison systématique avec scikit-learn pour valider la correction des implémentations
- **Analyses avancées** : Analyse des seuils optimaux, analyse des poids des features, benchmark de performance, expérimentations d'hyperparamètres
- **Tests unitaires** : Suite de 31 tests couvrant l'ensemble des modules du système

### Technologies Utilisées

- **Langage C** : Implémentation principale des algorithmes (GCC 12.5.0, standard C99)
- **Python 3** : Scripts d'analyse et visualisation (pandas, numpy, matplotlib, seaborn, scikit-learn)
- **Make** : Automatisation de la compilation et de l'exécution
- **Docker** : Environnement reproductible (optionnel)

---

## Architecture du Projet

Le projet suit une architecture modulaire qui sépare clairement les responsabilités de chaque composant, facilitant la maintenance, les tests et l'extension future.

### Structure des Répertoires

```
credit_risk_prediction/
├── c/                               # C core (pipeline, modèles, métriques, tests)
│   ├── src/                         # Code source C
│   │   ├── main.c                   # Point d'entrée - Orchestre le pipeline complet
│   │   ├── utils/                   # Utilitaires généraux
│   │   ├── data/                    # Gestion des données (chargement, split)
│   │   ├── preprocessing/           # Prétraitement (scaler, encoder)
│   │   ├── models/                  # Algorithmes ML (régression logistique, arbre de décision)
│   │   ├── evaluation/              # Métriques et évaluation
│   │   └── analysis/                # Analyses avancées et expérimentations
│   ├── tests/                       # Tests unitaires (31 tests)
│   ├── build/                       # Fichiers de compilation (générés)
│   └── Makefile
├── python/                          # Scripts Python (validation sklearn, visualisation)
│   ├── scripts/
│   │   ├── explore_data.py
│   │   ├── plot_results.py
│   │   ├── compare_with_sklearn.py
│   │   ├── analyze_missing_values.py
│   │   ├── detect_outliers.py
│   │   └── split_dataset.py
│   └── requirements.txt
├── data/                            # Source unique de vérité
│   ├── raw/                         # Données brutes
│   │   └── credit_risk_dataset.csv
│   ├── processed/                   # Données prétraitées (générées par C)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── scaler_params.txt
│   │   └── split_info.txt
│   └── stats/                       # Statistiques et analyses (générées)
├── results/                         # Outputs générés (benchmarks, métriques, figures)
│   ├── metrics/
│   │   ├── logistic_regression/
│   │   └── decision_tree/
│   ├── plots/
│   │   ├── csv/                     # Données CSV pour visualisation
│   │   ├── logistic_regression/
│   │   └── decision_tree/
│   └── models/                      # Modèles sauvegardés (générés)
│       ├── logistic_model.bin
│       └── decision_tree_model.bin
├── docs/
├── Makefile                         # Délègue à c/Makefile
├── Dockerfile
└── README.md
```

### Organisation Modulaire

#### Module `utils/` - Utilitaires Généraux
- **Rôle** : Fonctions utilitaires réutilisables dans tout le projet
- **Composants** :
  - `utils.c/.h` : Fonctions mathématiques, validation, helpers
  - `csv_parser.c/.h` : Parsing CSV avec gestion des types (numérique/catégoriel) et chargement numérique pour datasets prétraités
  - `memory_manager.c/.h` : Allocation mémoire sécurisée (détection d'erreurs)

#### Module `data/` - Gestion des Données
- **Rôle** : Chargement et préparation des données
- **Composants** :
  - `data_loader.c/.h` : Chargement CSV → structure Dataset
  - `data_splitter.c/.h` : Division train/test avec shuffle aléatoire. Génère et sauvegarde `train.csv` et `test.csv` dans `data/processed/`
  - `prepare_dataset.c/.h` : Préparation séparée du dataset (split + sauvegarde train/test)

#### Module `preprocessing/` - Prétraitement
- **Rôle** : Transformation des données pour l'entraînement
- **Composants** :
  - `preprocessing.c/.h` : Orchestration du pipeline de prétraitement
  - `scaler.c/.h` : Normalisation StandardScaler (fit sur train, transform sur test)
  - `encoder.c/.h` : Encodage catégoriel (Label Encoding)

#### Module `models/` - Algorithmes ML
- **Rôle** : Implémentation des algorithmes de machine learning
- **Composants** :
  - `logistic_regression.c/.h` : Régression logistique avec Gradient Descent
  - `decision_tree.c/.h` : Arbre de décision CART (Gini/Entropy)

#### Module `evaluation/` - Évaluation
- **Rôle** : Calcul et sauvegarde des métriques de performance
- **Composants** :
  - `metrics.c/.h` : Calcul de toutes les métriques (Accuracy, Precision, Recall, F1, AUC-ROC)
  - `confusion_matrix.c/.h` : Calcul et affichage de la matrice de confusion

#### Module `analysis/` - Analyses Avancées
- **Rôle** : Analyses approfondies et expérimentations sur les modèles
- **Composants** :
  - `threshold_analysis.c/.h` : Analyse des seuils optimaux de classification avec calcul des coûts métier
  - `feature_weights_analysis.c/.h` : Analyse des poids et importance des features pour la régression logistique
  - `learning_rate_experiment.c/.h` : Expérimentation systématique de différentes valeurs de learning rate
  - `iterations_experiment.c/.h` : Expérimentation du nombre d'itérations pour l'entraînement

### Flux de Données

Le pipeline suit un flux séquentiel bien défini :

```
1. Chargement (data_loader)
   └─> CSV brut → Dataset structuré

2. Prétraitement (preprocessing)
   ├─> Encodage catégoriel (encoder)
   ├─> Gestion valeurs manquantes (preprocessing)
   └─> Normalisation (scaler)

3. Division (data_splitter)
   └─> Dataset → Train (80%) + Test (20%)
       └─> Sauvegarde train.csv et test.csv dans data/processed/
           (Ces fichiers sont ensuite chargés par les scripts Python)

4. Entraînement (models)
   ├─> Régression Logistique (logistic_regression)
   └─> Arbre de Décision (decision_tree)

5. Évaluation (evaluation)
   ├─> Calcul métriques (metrics)
   ├─> Matrice de confusion (confusion_matrix)
   └─> Sauvegarde résultats → results/metrics/

6. Analyses avancées (analysis)
   ├─> Analyse des seuils (threshold_analysis)
   ├─> Analyse des poids (feature_weights_analysis)
   └─> Benchmark de performance

7. Visualisation (scripts Python)
   └─> Génération graphiques → results/plots/
```

---

## Installation

### Prérequis

- **Compilateur C** : GCC (version 7.5 ou supérieure)
- **Make** : Pour la compilation automatisée
- **Python 3** : Pour les scripts d'analyse (optionnel)
  - Bibliothèques requises : pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation des Dépendances Python

```bash
pip install -r python/requirements.txt
```

### Compilation

```bash
# Compiler le projet
make

# Vérifier la compilation
ls -l c/build/credit_risk_predictor
```

**Flags de compilation** : `-Wall -Wextra -O2 -std=c99 -lm`

---

## Utilisation

### Exécution du Programme Principal

```bash
# Exécuter le programme complet (depuis la racine du projet)
./c/build/credit_risk_predictor

# Ou utiliser make
make run
```

**Note importante** : Le programme C génère les fichiers `train.csv` et `test.csv` dans `data/processed/`. Ces fichiers sont ensuite utilisés par les scripts Python (notamment `compare_with_sklearn.py`). Il est donc recommandé d'exécuter le programme C en premier. Si les fichiers `train.csv` et `test.csv` n'existent pas, le script Python affichera un message d'erreur demandant d'exécuter le programme C d'abord.

### Commandes Makefile Principales

- `make` : Compile le projet
- `make clean` : Nettoie les fichiers de compilation, résultats et modèles
- `make run` : Compile et exécute le programme
- `make results` : Génère les résultats C (métriques, CSV, modèles)
- `make compare` : Génère la comparaison avec scikit-learn
- `make plots` : Génère les graphiques Python
- `make plots-all` : Génère tous les graphiques (C et Python)
- `make verify` : Vérifie que tous les fichiers sont générés
- `make regenerate` : Régénère tout (clean + compile + results + compare + plots-all + verify)

### Pipeline d'Entraînement

Le programme exécute automatiquement un workflow complet :

1. **Chargement des données** : Lecture du dataset depuis `data/raw/credit_risk_dataset.csv` ou chargement des datasets pré-traités depuis `data/processed/` si disponibles
2. **Encodage catégoriel** : Transformation des variables catégorielles selon le schéma de label encoding défini
3. **Gestion des valeurs manquantes** : Imputation par médiane pour les variables numériques
4. **Division des données** : Split train/test avec ratio 80/20 effectué uniquement par le programme C. Les fichiers `train.csv` et `test.csv` sont sauvegardés dans `data/processed/`. Si ces fichiers existent déjà, ils sont chargés directement. Sinon, le split est effectué et les fichiers sont générés. Le script Python `compare_with_sklearn.py` charge directement ces fichiers pour garantir l'utilisation des mêmes données et permettre une comparaison équitable avec scikit-learn
5. **Normalisation** : Application du StandardScaler (ajustement sur l'ensemble d'entraînement, transformation sur l'ensemble de test)
6. **Entraînement de la régression logistique** : Optimisation par descente de gradient (1000 itérations, taux d'apprentissage 0.01)
7. **Évaluation de la régression logistique** : Calcul des métriques sur les ensembles d'entraînement et de test
8. **Analyses avancées** : Génération automatique de l'analyse des seuils optimaux, de l'analyse des poids des features, du tableau de convergence et du benchmark de performance
9. **Entraînement de l'arbre de décision** : Construction de l'arbre avec max_depth=7 et critère d'impureté Gini
10. **Évaluation de l'arbre de décision** : Calcul des métriques sur les ensembles d'entraînement et de test
11. **Comparaison des modèles** : Analyse comparative des performances avec affichage des métriques principales
12. **Persistance** : Sauvegarde des modèles entraînés et des résultats d'évaluation
13. **Génération des graphiques** : Appel automatique de `plot_results.py` pour générer tous les graphiques

---

## Résultats et Performances

### Performances Obtenues

**Régression Logistique (Implémentation C)** :
- Accuracy : 79.96%
- AUC-ROC : 79.95%
- Precision : 48.27%
- Recall : 47.43%
- F1-Score : 47.84%

**Arbre de Décision (Implémentation C)** :
- Accuracy : 93.52%
- AUC-ROC : 91.94%
- Precision : 97.30%
- Recall : 68.49%
- F1-Score : 80.39%

### Comparaison avec Scikit-learn

Les implémentations C sont comparées systématiquement avec scikit-learn pour validation :

**Régression Logistique** :
- L'écart observé (Accuracy -4.96%, AUC-ROC -5.26%) est principalement dû à la différence d'algorithme d'optimisation (Gradient Descent vs L-BFGS)
- Le Recall est très proche (47.43% vs 48.17%), confirmant que les deux implémentations utilisent exactement les mêmes données train/test (générées par C)

**Arbre de Décision** :
- Résultats très similaires (différence < 5%) confirmant la correction de l'implémentation

### Benchmark de Performance Computationnelle

Temps d'exécution mesurés pour la régression logistique :

| Opération | Temps (secondes) | Pourcentage du total |
|-----------|-----------------|---------------------|
| Chargement CSV + encodage | ~0.100 | 23.1% |
| Prétraitement (imputation + shuffle) | ~0.030 | 6.9% |
| Normalisation (fit + transform) | ~0.003 | 0.7% |
| Entraînement (1000 itérations) | ~0.300 | 69.3% |
| Évaluation (prédictions + métriques) | ~0.003 | 0.7% |
| **TOTAL** | **~0.433** | **100%** |

**Arbre de décision** : Temps d'entraînement environ 4.62 secondes

---

## Structure des Fichiers Générés

### Répertoire `results/metrics/logistic_regression/`

**Métriques C** :
- `lr_c_train_metrics.txt` : Métriques d'entraînement (Accuracy, Precision, Recall, F1-Score)
- `lr_c_test_metrics.txt` : Métriques de test (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- `lr_c_test_confusion_matrix.txt` : Matrice de confusion (TN, FP, FN, TP)

**Analyses avancées C** :
- `lr_c_convergence_table.txt` : Tableau de convergence avec variations de coût par itération
- `lr_c_threshold_analysis.txt` : Analyse des seuils optimaux (0.3, 0.4, 0.5, 0.6, 0.7) avec calcul des coûts métier
- `lr_c_feature_weights_analysis.txt` : Analyse des poids et importance des features (classement par valeur absolue)
- `lr_c_performance_benchmark.txt` : Benchmark de performance computationnelle détaillé par étape

**Métriques Python (scikit-learn)** :
- `lr_python_train_metrics.txt` : Métriques d'entraînement Python
- `lr_python_test_metrics.txt` : Métriques de test Python
- `lr_python_test_confusion_matrix.txt` : Matrice de confusion Python

**Comparaison** :
- `lr_comparison_c_vs_python.txt` : Tableau comparatif détaillé des métriques C vs Python

### Répertoire `results/metrics/decision_tree/`

**Métriques C** :
- `dt_c_train_metrics.txt` : Métriques d'entraînement
- `dt_c_test_metrics.txt` : Métriques de test
- `dt_c_test_confusion_matrix.txt` : Matrice de confusion
- `dt_c_tree_stats.txt` : Statistiques de l'arbre (profondeur réelle, nombre de nœuds, temps d'entraînement)

**Métriques Python (scikit-learn)** :
- `dt_python_train_metrics.txt` : Métriques d'entraînement Python
- `dt_python_test_metrics.txt` : Métriques de test Python
- `dt_python_test_confusion_matrix.txt` : Matrice de confusion Python
- `dt_python_tree_stats.txt` : Statistiques de l'arbre Python

**Comparaison** :
- `dt_comparison_c_vs_python.txt` : Tableau comparatif détaillé des métriques C vs Python

### Répertoire `results/plots/`

**Graphiques Régression Logistique** :
- `lr_c_confusion_matrix_test.png` : Matrice de confusion visuelle (C)
- `lr_c_metrics_train_vs_test.png` : Comparaison métriques train vs test (C)
- `lr_c_cost_curve_training.png` : Courbe de coût pendant l'entraînement (C)
- `lr_python_confusion_matrix_test.png` : Matrice de confusion visuelle (Python)
- `lr_python_metrics_train_vs_test.png` : Comparaison métriques train vs test (Python)
- `lr_python_cost_curve_training.png` : Courbe de coût pendant l'entraînement (Python)
- `lr_dt_c_roc_curves_comparison.png` : Comparaison des courbes ROC LR vs DT (C)
- `lr_dt_python_roc_curves_comparison.png` : Comparaison des courbes ROC LR vs DT (Python)
- `summary_lr_c.png` : Figure récapitulative complète (C)
- `summary_lr_python.png` : Figure récapitulative complète (Python)

**Graphiques Arbre de Décision** :
- `dt_c_confusion_matrix_test.png` : Matrice de confusion visuelle (C)
- `dt_c_metrics_train_vs_test.png` : Comparaison métriques train vs test (C)
- `dt_c_feature_importance.png` : Importance des features (C)
- `dt_python_confusion_matrix_test.png` : Matrice de confusion visuelle (Python)
- `dt_python_metrics_train_vs_test.png` : Comparaison métriques train vs test (Python)
- `summary_dt_c.png` : Figure récapitulative complète (C)
- `summary_dt_python.png` : Figure récapitulative complète (Python)

**Figure globale** :
- `summary_figure.png` : Vue d'ensemble générale comparant tous les modèles

**Données pour visualisation** (`results/plots/csv/`) :
- `lr_c_cost_curve.csv` : Données de la courbe de coût (C)
- `lr_python_cost_curve.csv` : Données de la courbe de coût (Python)
- `lr_roc_data.csv` : Données pour courbe ROC (C)
- `lr_python_roc_data.csv` : Données pour courbe ROC (Python)
- `dt_roc_data.csv` : Données pour courbe ROC arbre de décision (C)
- `dt_python_roc_data.csv` : Données pour courbe ROC arbre de décision (Python)

**Total** : 17 graphiques PNG générés automatiquement

---

## Documentation

### Documentation API

Pour comprendre en détail l'utilisation de chaque fonction :
- `docs/api/functions_documentation.md` : Documentation complète de toutes les fonctions avec paramètres, valeurs de retour et exemples

### Documentation du Code Source

Tous les fichiers source du projet incluent une documentation complète et standardisée :
- **En-têtes professionnels** : Chaque fichier commence par un en-tête avec nom, rôle, auteur, version et instructions d'usage
- **Documentation des fonctions** : Toutes les fonctions sont documentées selon un format standardisé avec rôle, paramètres et valeur de retour
- **Encadrés visuels** : Les sections importantes du code sont marquées par des encadrés visuels clairs pour faciliter la navigation

### Rapports Techniques

Le projet inclut une documentation académique complète :
- **Rapport principal** : `docs/rapport/M1 - TAA - Maxime BRONNY - 19009314.pdf`
- **Présentation** : `docs/presentation/slides.pdf`

---

## Auteur

**Projet M1** - Techniques d'Apprentissage Artificiel  
**Auteur** : Maxime BRONNY  
**Année Académique** : 2025-2026

---

## Références

- Kaggle Credit Risk Dataset
- Breiman, L. (1984). Classification and Regression Trees (CART)
- Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression
- Andrew Ng - Machine Learning Course (Coursera)
- Bishop, C. (2006). Pattern Recognition and Machine Learning
- Scikit-learn Documentation

---

**Dernière mise à jour** : Décembre 2025
