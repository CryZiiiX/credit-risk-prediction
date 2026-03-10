# Configuration Docker - Prédiction du Risque de Crédit Bancaire

> **Guide d'installation et d'utilisation de Docker pour le projet**  
> Ce document explique comment installer Docker et utiliser l'environnement conteneurisé pour garantir la reproductibilité du projet sur toutes les machines.

---

## Installation de Docker

Si Docker n'est pas encore installé sur votre système, suivez ces étapes :

### 1. Installer Docker

```bash
# Mettre à jour les paquets
sudo apt update

# Installer Docker
sudo apt install -y docker.io

# Démarrer et activer Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. Configurer les permissions (optionnel mais recommandé)

Pour éviter d'utiliser `sudo` à chaque commande Docker :

```bash
# Ajouter votre utilisateur au groupe docker
sudo usermod -aG docker $USER

# Appliquer les changements (ou se déconnecter/reconnecter)
newgrp docker
```

### 3. Vérifier l'installation

```bash
# Vérifier la version de Docker
docker --version

# Tester avec l'image hello-world
docker run hello-world
```

## Utilisation avec le Projet

Une fois Docker installé et configuré :

### Build de l'image

```bash
# Se placer dans le répertoire du projet
cd <chemin-vers-le-projet>

# Build l'image
docker build -t credit-risk-predictor:latest .
```

**Durée du build** : Environ 3-5 minutes la première fois (téléchargement de l'image de base `gcc:12-bookworm` + installation des dépendances Python).

**Contenu de l'image** :
- GCC 12.5.0 (compilateur C)
- Python 3.11.2 + pip
- Toutes les dépendances Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Le projet compilé avec :
  - Les deux modèles ML (Régression Logistique et Arbre de Décision)
  - Les modules d'analyse (threshold_analysis, feature_weights_analysis)
  - Les expérimentations (learning_rate_experiment, iterations_experiment)
  - Le module de préparation (prepare_dataset)

### Exécution

#### Exécution simple

```bash
# Exécution du programme C complet (entraîne les deux modèles)
docker run --rm credit-risk-predictor:latest
```

Cette commande entraîne automatiquement :
- **Régression Logistique** (implémentation C)
- **Arbre de Décision** (implémentation C)

#### Exécution avec sauvegarde des résultats

```bash
# Sauvegarder les résultats et modèles sur votre machine locale
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest
```

**Résultats générés** :
- `results/models/logistic_model.bin` : Modèle de régression logistique sauvegardé
- `results/models/decision_tree_model.bin` : Modèle d'arbre de décision sauvegardé
- `data/processed/train.csv` : Dataset d'entraînement prétraité
- `data/processed/test.csv` : Dataset de test prétraité
- `data/processed/scaler_params.txt` : Paramètres du StandardScaler
- `results/metrics/logistic_regression/` : Métriques et analyses pour LR
  - `lr_c_train_metrics.txt`, `lr_c_test_metrics.txt` : Métriques train/test
  - `lr_c_test_confusion_matrix.txt` : Matrice de confusion
  - `lr_c_convergence_table.txt` : Tableau de convergence avec variations
  - `lr_c_threshold_analysis.txt` : Analyse des seuils optimaux
  - `lr_c_feature_weights_analysis.txt` : Analyse des poids des features
  - `lr_c_performance_benchmark.txt` : Benchmark de performance computationnelle
- `results/metrics/decision_tree/` : Métriques train/test pour DT
- `results/plots/csv/` : Données CSV pour courbes ROC et cost curves

#### Préparation du dataset

```bash
# Générer le split train/test et sauvegarder les fichiers nécessaires
docker run --rm \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest ./c/build/prepare_dataset
```

Cette commande génère les datasets prétraités et les paramètres du scaler, permettant de réutiliser le même split pour plusieurs expériences.

**Fichiers générés** :
- `data/processed/train.csv` : Dataset d'entraînement prétraité (non normalisé)
- `data/processed/test.csv` : Dataset de test prétraité (non normalisé)
- `data/processed/scaler_params.txt` : Paramètres du StandardScaler
- `data/processed/split_info.txt` : Métadonnées sur le split

#### Nouvelles fonctionnalités d'analyse

##### Expérimentation du Learning Rate

```bash
# Tester différents learning rates (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest ./c/build/learning_rate_experiment
```

**Fichier généré** : `results/metrics/logistic_regression/lr_c_learning_rate_experiment.txt`

Cette expérimentation teste 6 valeurs de learning rate et génère un rapport détaillé avec :
- État de convergence pour chaque learning rate
- Accuracy et F1-Score sur le test set
- Temps d'entraînement pour chaque configuration

##### Expérimentation du Nombre d'Itérations

```bash
# Tester différents nombres d'itérations (100, 500, 1000, 2000, 5000)
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest ./c/build/iterations_experiment
```

**Fichier généré** : `results/metrics/logistic_regression/lr_c_iterations_experiment.txt`

Cette expérimentation teste 5 configurations d'itérations et génère un rapport avec :
- État de convergence selon le nombre d'itérations
- Évolution de l'accuracy et du F1-Score
- Impact du nombre d'itérations sur le temps de calcul

#### Exécuter les tests unitaires

```bash
# Exécuter tous les tests (31 tests au total)
docker run --rm credit-risk-predictor:latest bash -c "cd c/tests && bash run_tests.sh"
```

#### Comparaison avec scikit-learn

```bash
# Générer les modèles Python de référence et comparer avec l'implémentation C
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest python3 python/scripts/compare_with_sklearn.py
```

Cette commande génère :
- Métriques Python pour les deux modèles (LR et DT)
- Fichiers de comparaison C vs Python
- Données pour les graphiques Python

#### Générer les graphiques de visualisation

```bash
# Générer tous les graphiques (17 graphiques au total)
docker run --rm \
  -v $(pwd)/results:/app/results \
  credit-risk-predictor:latest python3 python/scripts/plot_results.py
```

**Graphiques générés** :
- `results/plots/summary_figure.png` : Vue d'ensemble générale
- `results/plots/logistic_regression/` : 7 graphiques pour LR (C et Python)
- `results/plots/decision_tree/` : 6 graphiques pour DT (C et Python)
- `results/plots/csv/` : Données CSV pour visualisation

#### Réutiliser les résultats existants (sans ré-entraînement)

Si vous avez déjà exécuté le programme et souhaitez régénérer uniquement les graphiques ou la comparaison sklearn sans ré-entraîner les modèles C :

```bash
# Générer les graphiques à partir des métriques existantes
docker run --rm \
  -v $(pwd)/results:/app/results \
  credit-risk-predictor:latest python3 python/scripts/plot_results.py

# Comparer avec scikit-learn (charge les métriques C existantes, entraîne sklearn pour comparaison)
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest python3 python/scripts/compare_with_sklearn.py
```

Les modèles C (`results/models/*.bin`) et les métriques (`results/metrics/`) sont réutilisés tels quels.

#### Mode interactif (exploration)

```bash
# Lancer un shell interactif dans le container
docker run --rm -it credit-risk-predictor:latest bash
```

Dans le shell, vous pouvez :
- Explorer les fichiers générés
- Exécuter des commandes manuelles
- Tester des modifications

## Dépannage

### Erreur "Permission denied"

Si vous obtenez une erreur de permission lors de l'utilisation de Docker :

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Nettoyage de l'espace disque

Si Docker occupe trop d'espace :

```bash
# Supprimer les images et containers inutilisés
docker system prune -a

# Vérifier l'espace disque disponible
df -h
```

### Rebuild complet

Si vous modifiez le code et souhaitez reconstruire l'image :

```bash
# Supprimer l'ancienne image
docker rmi credit-risk-predictor:latest

# Rebuild
docker build -t credit-risk-predictor:latest .
```

## Versions et Compatibilité

Pour connaître les versions exactes de tous les composants (GCC 12.5.0, Python 3.11.2, bibliothèques scientifiques), consultez le fichier **`VERSIONS.md`** qui documente en détail :
- Les versions de tous les composants
- La matrice de compatibilité
- Les commandes de vérification
- Les notes techniques

## Pour le Correcteur/Professeur

Si vous êtes un correcteur et souhaitez tester le projet rapidement :

```bash
# 1. Extraire l'archive ou cloner le dépôt
cd credit-risk-project

# 2. Build l'image (une seule fois, ~3-5 minutes)
docker build -t credit-risk-predictor:latest .

# 3. Exécuter le programme complet (entraîne les deux modèles)
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest

# 4. (Optionnel) Générer les graphiques (réutilise les résultats existants)
docker run --rm \
  -v $(pwd)/results:/app/results \
  credit-risk-predictor:latest python3 python/scripts/plot_results.py

# 5. (Optionnel) Exécuter les tests unitaires
docker run --rm credit-risk-predictor:latest bash -c "cd c/tests && bash run_tests.sh"

# 6. (Optionnel) Comparaison avec scikit-learn
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest python3 python/scripts/compare_with_sklearn.py

# 7. (Optionnel) Expérimentations d'hyperparamètres
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest ./c/build/learning_rate_experiment

docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  credit-risk-predictor:latest ./c/build/iterations_experiment
```

**Temps total estimé** :
- Build initial : ~3-5 minutes
- Exécution du programme : ~5-10 secondes
- Génération des graphiques : ~10-15 secondes
- Tests unitaires : ~2-3 secondes
- Comparaison scikit-learn : ~5-10 secondes
- Expérimentation learning rate : ~2-3 secondes
- Expérimentation itérations : ~8-10 secondes

**Total** : ~6-8 minutes pour un test complet avec toutes les fonctionnalités

## Résultats Attendus

Lors de l'exécution du programme, vous devriez voir :

### Compilation

- Compilation sans erreurs ni warnings
- Tous les fichiers sources compilés :
  - Modèles ML : `logistic_regression.c`, `decision_tree.c`
  - Modules d'analyse : `threshold_analysis.c`, `feature_weights_analysis.c`
  - Expérimentations : `learning_rate_experiment.c`, `iterations_experiment.c`
  - Utilitaires : `prepare_dataset.c`
  - Modules de base : `data_loader.c`, `data_splitter.c`, `scaler.c`, `encoder.c`, `metrics.c`, etc.

### Exécution

- Chargement du dataset (32 581 lignes)
- Prétraitement des données (encodage catégoriel, normalisation)
- Division train/test (80/20)

### Entraînement des Modèles

- **Régression Logistique** : Entraînement (~0.4s)
- **Arbre de Décision** : Entraînement (~4.6s, profondeur=7, 79 nœuds)

### Métriques de Performance

**Régression Logistique (Implémentation C)** :
- Accuracy : **81.17%**
- Precision : ~79.28%
- Recall : ~50.77%
- F1-Score : ~61.89%
- AUC-ROC : **82.20%**

**Arbre de Décision (Implémentation C)** :
- Accuracy : **93.48%**
- Precision : ~79.28%
- Recall : ~91.04%
- F1-Score : ~84.82%
- AUC-ROC : **91.04%**
- Profondeur réelle : 7
- Nombre de nœuds : 79
- Temps d'entraînement : ~4.6s

### Fichiers Générés

**Modèles** :
- `results/models/logistic_model.bin` : Modèle de régression logistique
- `results/models/decision_tree_model.bin` : Modèle d'arbre de décision

**Datasets prétraités** :
- `data/processed/train.csv` : Dataset d'entraînement prétraité
- `data/processed/test.csv` : Dataset de test prétraité
- `data/processed/scaler_params.txt` : Paramètres du StandardScaler
- `data/processed/split_info.txt` : Métadonnées sur le split

**Métriques et analyses - Régression Logistique** :
- `results/metrics/logistic_regression/lr_c_train_metrics.txt` : Métriques d'entraînement
- `results/metrics/logistic_regression/lr_c_test_metrics.txt` : Métriques de test
- `results/metrics/logistic_regression/lr_c_test_confusion_matrix.txt` : Matrice de confusion
- `results/metrics/logistic_regression/lr_c_convergence_table.txt` : Tableau de convergence avec variations
- `results/metrics/logistic_regression/lr_c_threshold_analysis.txt` : Analyse des seuils optimaux
- `results/metrics/logistic_regression/lr_c_feature_weights_analysis.txt` : Analyse des poids des features
- `results/metrics/logistic_regression/lr_c_performance_benchmark.txt` : Benchmark de performance

**Métriques - Arbre de Décision** :
- `results/metrics/decision_tree/dt_c_train_metrics.txt` : Métriques d'entraînement
- `results/metrics/decision_tree/dt_c_test_metrics.txt` : Métriques de test
- `results/metrics/decision_tree/dt_c_test_confusion_matrix.txt` : Matrice de confusion
- `results/metrics/decision_tree/dt_c_tree_stats.txt` : Statistiques de l'arbre (profondeur, nœuds, temps)

**Graphiques** (après exécution de `plot_results.py`) :
- 17 graphiques PNG dans `results/plots/`
- Données CSV pour courbes ROC dans `results/plots/csv/`

## Avantages de cette Approche

- **Reproductibilité** : Même environnement sur toutes les machines
- **Isolation** : Pas d'interférence avec les installations système
- **Simplicité** : Une seule commande pour exécuter tout le projet
- **Professionnalisme** : Démontre la maîtrise des outils modernes de développement

