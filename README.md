<!--
======================================================================================
Nom     : README.md
Rôle    : Document de présentation, d'installation et d'exécution du projet.
Auteur  : Maxime BRONNY
Version : V1
Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
          Master 1 Informatique Big Data
Usage   : Document de présentation, d'installation et d'exécution du projet.
======================================================================================
-->

# Prédiction du Risque de Crédit Bancaire

**Projet M1 Informatique - Techniques d'Apprentissage Artificiel**
**Auteur : Maxime BRONNY - Université Paris 8**

Pipeline complet d'apprentissage automatique en **Python**, avec trois algorithmes de
classification implémentés **from scratch** (numpy uniquement) : régression logistique,
arbre de décision CART et k plus proches voisins. scikit-learn n'est utilisé que pour
**vérifier** la justesse des implémentations.

> ℹ️ La première version de ce projet était implémentée en C. Ce code et tous les
> fichiers associés (anciens rapports, résultats, Docker C) sont conservés à titre
> d'historique en dehors du dossier de rendu (dans le dossier parent
> `../archive_obsolete/`), mais ne sont plus utilisés : toute la partie
> algorithmique est désormais en Python, conformément aux consignes de la matière.

---

## 1. Contexte et problématique

Lorsqu'une banque accorde un crédit, elle prend le risque que l'emprunteur ne rembourse
pas. **Problématique** : à partir des caractéristiques d'un emprunteur et de son prêt
(âge, revenu, montant, taux, historique...), peut-on prédire automatiquement le risque
de défaut de paiement, et quelle méthode d'apprentissage supervisé offre le meilleur
compromis entre détection des défauts et erreurs de classification ?

Il s'agit d'une **classification binaire supervisée** :
- classe 0 : remboursement normal (~78 % des cas) ;
- classe 1 : défaut de paiement (~22 % des cas).

Le déséquilibre des classes est un enjeu central du projet : un modèle naïf qui prédit
toujours « pas de défaut » atteint déjà 78 % d'accuracy, d'où l'importance des métriques
precision / recall / F1 / AUC.

## 2. Dataset

**Credit Risk Dataset** (Kaggle) - `data/raw/credit_risk_dataset.csv`
- 32 581 emprunteurs, 11 variables explicatives + 1 cible (`loan_status`) ;
- 8 variables numériques (âge, revenu, ancienneté d'emploi, montant, taux, etc.) ;
- 4 variables catégorielles (type de logement, objet du prêt, grade A-G, antécédent de
  défaut) ;
- valeurs manquantes : `person_emp_length` (~2,7 %) et `loan_int_rate` (~9,6 %) ;
- quelques valeurs aberrantes (âges > 120 ans) supprimées au nettoyage (7 lignes).

## 3. Architecture du projet

```text
PROGRAMME/
├── main.py                 # Point d'entrée : python3 main.py
├── requirements.txt        # Dépendances Python
├── Makefile                # make run / make test / make clean
├── Dockerfile              # Image reproductible (docker build / docker run)
├── src/                    # Code source du pipeline
│   ├── data_loader.py      # Chargement + validation du CSV
│   ├── preprocessing.py    # Nettoyage, encodage, imputation, normalisation
│   ├── split_data.py       # Split stratifié train/val/test (70/15/15)
│   ├── metrics.py          # Métriques from scratch (accuracy → AUC-ROC)
│   ├── visualization.py    # Figures du rapport
│   └── models/             # Les 3 algorithmes from scratch
├── tests/                  # Tests pytest (11 tests)
├── data/
│   ├── raw/                # Dataset brut (inchangé)
│   ├── splits/             # train.csv / val.csv / test.csv (générés)
│   └── stats/              # Analyse exploratoire (figures, stats)
├── results/
│   ├── metrics.json        # Toutes les métriques (3 modèles × train/test)
│   ├── predictions.csv     # Prédictions et probabilités sur le test
│   ├── sklearn_comparison.json  # Vérification from scratch vs sklearn
│   └── models/*.pkl        # Modèles entraînés
├── reports/
│   ├── figures/            # 7 figures générées pour le rapport
│   └── latex/              # Rapport LaTeX (main.tex, sections/, bibliographie)
└── docs/
    └── rapport/            # Rapport final au format PDF
```

> L'ancienne version C et les sauvegardes intermédiaires sont conservées en dehors
> de ce dossier (dans le dossier parent `PROJET/`) afin de ne garder ici que les
> fichiers utiles au rendu final.

📄 **Rapport** : le rapport final au format PDF se trouve dans `docs/rapport/` ;
ses sources LaTeX (compilables) sont dans `reports/latex/`.

## 4. Algorithmes implémentés (from scratch)

| Algorithme | Fichier | Principe | Hyperparamètres |
|---|---|---|---|
| Régression logistique | `src/models/logistic_regression.py` | sigmoïde(w·x+b), descente de gradient sur la log-loss | lr=0.1, 1000 itérations, seuil choisi sur validation |
| Arbre de décision CART | `src/models/decision_tree.py` | partitionnement récursif minimisant l'impureté de Gini (recherche de seuils vectorisée) | profondeur choisie sur validation parmi {3,5,7,9} |
| k-NN | `src/models/knn.py` | vote des k plus proches voisins (distance euclidienne, calcul par chunks) | k choisi sur validation parmi {5,11,21,31} |

Trois familles complémentaires : un modèle **linéaire** interprétable, un modèle
**non linéaire** à base de règles, et une méthode **non paramétrique** à base
d'instances.

## 5. Découpage des données

**70 % entraînement / 15 % validation / 15 % test**, split **stratifié** (le taux de
défaut de 21,8 % est préservé dans chaque ensemble), graine fixe (42) pour la
reproductibilité.

- l'**entraînement** sert à apprendre les paramètres des modèles ;
- la **validation** sert uniquement à choisir les hyperparamètres (profondeur de
  l'arbre, k du k-NN, seuils de décision) - critère : F1-score ;
- le **test** n'est utilisé qu'une seule fois, à la fin, pour mesurer la performance de
  généralisation réelle.

Ce découpage en trois ensembles évite le biais classique qui consiste à régler les
hyperparamètres sur le test (ce qui surestimerait les performances).

## 6. Métriques

Accuracy, précision, rappel, F1-score, matrice de confusion, courbe ROC et AUC -
toutes implémentées from scratch dans `src/metrics.py` et sauvegardées dans
`results/metrics.json`. Dans le contexte bancaire, le **faux négatif** (défaut non
détecté → crédit accordé à tort) est l'erreur la plus coûteuse, d'où l'attention portée
au rappel et au F1 plutôt qu'à la seule accuracy.

## 7. Installation

```bash
# Python 3.9+ requis
pip install -r requirements.txt
```

(Sur Ubuntu récent, si pip refuse : `pip install --user --break-system-packages -r requirements.txt`
ou utilisez un environnement virtuel `python3 -m venv .venv && source .venv/bin/activate`.)

## 8. Exécution

```bash
python3 main.py                    # pipeline complet (~20 s)
python3 main.py --compare-sklearn  # + vérification avec scikit-learn
python3 -m pytest tests/ -v       # tests (11 tests)
# ou : make run / make compare / make test
```

**Avec Docker (optionnel - exécution reproductible sans installer les dépendances) :**

```bash
docker build -t credit-risk .   # construit l'image (lance aussi les 11 tests)
docker run --rm credit-risk     # exécute le pipeline complet (python3 main.py)
```

## 9. Résultats (ensemble de test, split 70/15/15, seed 42)

| Modèle | Accuracy | Précision | Rappel | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| Régression logistique | 0,820 | 0,573 | 0,690 | 0,626 | 0,848 |
| **Arbre de décision (depth=7)** | **0,931** | **0,977** | **0,703** | **0,817** | **0,909** |
| k-NN (k=31) | 0,862 | 0,683 | 0,684 | 0,683 | 0,877 |

L'arbre de décision domine sur toutes les métriques : le problème contient des
interactions non linéaires (ex. ratio prêt/revenu × grade) qu'un modèle linéaire ne
capture pas. Le k-NN se place entre les deux.

**Vérification scikit-learn** (mêmes données, mêmes hyperparamètres) : écart d'AUC
≤ 0,0005 pour les trois modèles - les implémentations from scratch sont validées
(`results/sklearn_comparison.json`).

## 10. Limites

- L'encodage label des variables nominales introduit un ordre artificiel (acceptable
  pour l'arbre et le k-NN, plus discutable pour la régression logistique) ;
- le déséquilibre des classes est géré par le réglage du seuil, pas par
  ré-échantillonnage (SMOTE, pondération) ;
- pas de validation croisée (un seul split, même si la graine est fixe) ;
- la recherche d'hyperparamètres reste volontairement réduite (petites grilles).

## 11. Pistes d'amélioration

- One-hot encoding des variables nominales et régularisation L2 pour la régression ;
- random forest (bagging d'arbres) pour réduire la variance de l'arbre seul ;
- validation croisée k-fold pour des estimations plus robustes ;
- analyse coût-bénéfice métier (coût asymétrique FN/FP) pour le choix final du seuil.

---

**Année académique 2025-2026 - dernière mise à jour : juin 2026**
