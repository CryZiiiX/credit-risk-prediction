# Paramètres des Modèles - Régression Logistique et Arbre de Décision

**Document synthétique des hyperparamètres utilisés dans les implémentations C et Python**

---

## 1. Régression Logistique

### 1.1. Implémentation C (LR_C)

| Paramètre | Valeur | Type | Rôle |
|-----------|--------|------|------|
| **learning_rate** | `0.01` | `double` | Taux d'apprentissage pour la descente de gradient. Contrôle la taille des pas lors de la mise à jour des poids. Valeur trop élevée → oscillations/divergence, trop faible → convergence lente. |
| **max_iterations** | `1000` | `int` | Nombre maximum d'itérations de la descente de gradient. Limite le temps d'entraînement et évite les boucles infinies. Le modèle converge généralement avant d'atteindre cette limite. |
| **threshold** | `0.5` | `double` | Seuil de décision pour la classification binaire. Si la probabilité prédite ≥ 0.5 → classe 1, sinon → classe 0. Peut être ajusté pour optimiser Precision/Recall. |

**Structure du modèle** :
- `weights` : Tableau de poids (un par feature)
- `bias` : Biais (terme constant)
- `n_features` : Nombre de features (déterminé automatiquement)

**Algorithme d'optimisation** : Gradient Descent (descente de gradient)

**Fonction de coût** : Cross-Entropy Loss (entropie croisée)

---

### 1.2. Implémentation Python (LR_Python)

| Paramètre | Valeur | Type | Rôle |
|-----------|--------|------|------|
| **solver** | `'lbfgs'` | `str` | Algorithme d'optimisation. L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) est une méthode quasi-Newton plus efficace que Gradient Descent. Converge généralement plus vite avec moins d'itérations. |
| **max_iter** | `1000` | `int` | Nombre maximum d'itérations pour l'algorithme L-BFGS. Correspond à max_iterations de l'implémentation C pour comparaison équitable. |
| **random_state** | `42` | `int` | Graine aléatoire pour la reproductibilité. Initialise les poids de manière déterministe. Non utilisé dans l'implémentation C (poids initialisés à zéro). |

**Paramètres par défaut (non explicitement définis)** :
- `penalty='l2'` : Régularisation L2 (par défaut)
- `C=1.0` : Inverse de la force de régularisation (par défaut)
- `tol=1e-4` : Tolérance pour le critère d'arrêt (par défaut)

**Algorithme d'optimisation** : L-BFGS (méthode quasi-Newton)

**Fonction de coût** : Cross-Entropy Loss avec régularisation L2 optionnelle

---

## 2. Arbre de Décision

### 2.1. Implémentation C (DT_C)

| Paramètre | Valeur | Type | Rôle |
|-----------|--------|------|------|
| **max_depth** | `7` | `int` | Profondeur maximale de l'arbre. Limite la complexité et prévient l'overfitting. Un arbre trop profond mémorise les données d'entraînement. |
| **min_samples_split** | `20` | `int` | Nombre minimum d'échantillons requis pour diviser un nœud interne. Empêche la création de divisions sur de très petits groupes, réduisant l'overfitting. |
| **min_samples_leaf** | `10` | `int` | Nombre minimum d'échantillons requis dans une feuille. Garantit que chaque feuille contient suffisamment d'échantillons pour des prédictions fiables. |
| **criterion** | `GINI` | `enum` | Critère d'impureté pour évaluer la qualité d'une division. Gini mesure l'impureté : 0 = pur (une seule classe), 1 = maximum d'impureté (classes équilibrées). Alternative : ENTROPY. |

**Structure du modèle** :
- `root` : Nœud racine de l'arbre
- `n_features` : Nombre de features (déterminé automatiquement)

**Algorithme** : CART (Classification and Regression Trees)

**Critères d'arrêt** :
- Profondeur atteinte (`depth >= max_depth`)
- Impureté nulle (`impurity == 0.0`)
- Gain de division ≤ 0
- Nombre d'échantillons insuffisant pour diviser ou créer une feuille

---

### 2.2. Implémentation Python (DT_Python)

| Paramètre | Valeur | Type | Rôle |
|-----------|--------|------|------|
| **max_depth** | `7` | `int` | Profondeur maximale de l'arbre. Identique à l'implémentation C pour comparaison équitable. |
| **min_samples_split** | `20` | `int` | Nombre minimum d'échantillons requis pour diviser un nœud. Identique à l'implémentation C. |
| **min_samples_leaf** | `10` | `int` | Nombre minimum d'échantillons requis dans une feuille. Identique à l'implémentation C. |
| **criterion** | `'gini'` | `str` | Critère d'impureté. 'gini' correspond à GINI dans l'implémentation C. Alternative : 'entropy'. |
| **random_state** | `42` | `int` | Graine aléatoire pour la reproductibilité. Utilisé pour le choix aléatoire des features lors de la division (si max_features < n_features). Non utilisé dans l'implémentation C (toutes les features sont évaluées). |

**Paramètres par défaut (non explicitement définis)** :
- `splitter='best'` : Stratégie de division (meilleure division)
- `max_features=None` : Nombre maximum de features à considérer (toutes par défaut)
- `min_weight_fraction_leaf=0.0` : Fraction minimale de poids dans une feuille

**Algorithme** : CART (Classification and Regression Trees)

---

## 3. Comparaison des Paramètres

### 3.1. Régression Logistique : C vs Python

| Aspect | LR_C | LR_Python |
|--------|------|-----------|
| **Algorithme** | Gradient Descent | L-BFGS |
| **Learning Rate** | 0.01 (explicite) | N/A (géré par L-BFGS) |
| **Itérations** | 1000 | 1000 |
| **Initialisation** | Poids à zéro | Aléatoire (random_state=42) |
| **Régularisation** | Aucune | L2 (par défaut, C=1.0) |
| **Convergence** | Plus lente | Plus rapide (méthode quasi-Newton) |

**Impact sur les performances** :
- L-BFGS converge généralement plus vite et peut atteindre un meilleur optimum
- La régularisation L2 de scikit-learn peut légèrement réduire l'overfitting
- L'initialisation aléatoire peut donner des résultats légèrement différents à chaque exécution (sans random_state)

### 3.2. Arbre de Décision : C vs Python

| Aspect | DT_C | DT_Python |
|--------|------|-----------|
| **Hyperparamètres** | Identiques | Identiques |
| **Critère** | GINI | 'gini' (équivalent) |
| **Reproductibilité** | Déterministe | Déterministe (random_state=42) |
| **Algorithme** | CART | CART |

**Impact sur les performances** :
- Les hyperparamètres étant identiques, les résultats devraient être très similaires
- Les petites différences peuvent venir de l'implémentation (arrondis, ordre d'évaluation des features)

---

## 4. Paramètres de Prédiction

### 4.1. Seuil de Classification (Régression Logistique)

| Implémentation | Seuil par défaut | Rôle |
|----------------|------------------|------|
| **LR_C** | `0.5` | Seuil fixe pour `predict()`. Les probabilités ≥ 0.5 sont classées comme classe 1. |
| **LR_Python** | `0.5` | Seuil fixe pour `predict()`. Les probabilités ≥ 0.5 sont classées comme classe 1. |

**Note** : Les deux implémentations utilisent le même seuil par défaut. L'analyse des seuils optimaux (threshold_analysis) permet de trouver un seuil personnalisé pour optimiser Precision/Recall selon le contexte métier.

---

## 5. Paramètres de Prétraitement (Communs)

Les paramètres suivants sont utilisés de manière identique dans les deux implémentations :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| **Test Size** | `0.2` (20%) | Proportion du dataset utilisée pour le test. 80% pour l'entraînement, 20% pour le test. |
| **Random State** | `42` | Graine aléatoire pour le split train/test (reproductibilité). |
| **StandardScaler** | Fit sur train, transform sur test | Normalisation des features (moyenne=0, écart-type=1). |

---

## 6. Résumé des Valeurs par Modèle

### LR_C
```
learning_rate = 0.01
max_iterations = 1000
threshold = 0.5
```

### LR_Python
```
solver = 'lbfgs'
max_iter = 1000
random_state = 42
```

### DT_C
```
max_depth = 7
min_samples_split = 20
min_samples_leaf = 10
criterion = GINI
```

### DT_Python
```
max_depth = 7
min_samples_split = 20
min_samples_leaf = 10
criterion = 'gini'
random_state = 42
```

---

**Dernière mise à jour** : Décembre 2025

