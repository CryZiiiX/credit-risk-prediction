# ======================================================================================
# Nom     : src/models/decision_tree.py
# Rôle    : Implémente l'arbre de décision binaire CART utilisé dans le projet.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé pour entraîner et évaluer le modèle associé.
# ======================================================================================

"""Arbre de décision binaire CART pour la classification, from scratch.

Principe : l'arbre partitionne récursivement les données. À chaque nœud,
on cherche le couple (variable, seuil) qui minimise l'impureté de Gini
pondérée des deux enfants. Une feuille stocke la proportion de défauts
des exemples qui y tombent (= probabilité prédite).

La recherche du meilleur seuil est vectorisée avec numpy (tri + sommes
cumulées) : on évalue tous les seuils candidats d'une variable en une
seule passe, ce qui rend l'entraînement rapide même sur ~22 000 lignes.
"""

import numpy as np


class _Node:
    """Nœud de l'arbre. Soit interne (feature/threshold), soit feuille (proba)."""

    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.proba = None  # proportion de classe 1 si feuille


class DecisionTreeClassifier:
    """Arbre CART (critère de Gini) pour classification binaire.

    Hyperparamètres :
    - max_depth : profondeur maximale (contrôle principal du sur-apprentissage,
      choisi sur l'ensemble de validation) ;
    - min_samples_split : taille minimale d'un nœud pour tenter une division ;
    - min_samples_leaf : taille minimale d'une feuille (évite les feuilles
      construites sur quelques exemples, donc peu fiables).
    """

    def __init__(self, max_depth: int = 7, min_samples_split: int = 20,
                 min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_nodes = 0
        self.actual_depth = 0

    # ---------------------------------------------------------------------------
    # Entraînement
    # ---------------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Entraîne l'arbre (construction récursive via _build) et réinitialise les
        compteurs (nombre de nœuds, profondeur atteinte)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_nodes = 0
        self.actual_depth = 0
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Construit récursivement le sous-arbre des exemples (X, y).

        Le nœud devient une feuille si la profondeur maximale est atteinte, s'il
        contient trop peu d'exemples, ou s'il est déjà pur ; sinon on cherche la
        meilleure coupure et on construit les deux enfants.
        """
        node = _Node()
        self.n_nodes += 1
        self.actual_depth = max(self.actual_depth, depth)
        node.proba = float(y.mean())

        # Conditions d'arrêt : profondeur max, nœud trop petit ou pur.
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or node.proba in (0.0, 1.0)):
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:  # aucune division n'améliore l'impureté
            return node

        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Cherche le couple (variable, seuil) minimisant le Gini pondéré des deux
        groupes ; renvoie (None, None) si aucune coupure n'améliore l'impureté.

        Pour chaque variable, les valeurs triées et les sommes cumulées des classes
        donnent le Gini de tous les seuils candidats en une passe (Gini = 2p(1-p)).
        """
        n = len(y)
        parent_gini = 2 * y.mean() * (1 - y.mean())
        best_gini = parent_gini - 1e-12  # exiger une amélioration stricte
        best_feature, best_threshold = None, None

        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]

            n_left = np.arange(1, n)          # taille du groupe gauche
            n_right = n - n_left
            pos_left = np.cumsum(ys)[:-1]     # nb de positifs à gauche

            p_left = pos_left / n_left
            p_right = (ys.sum() - pos_left) / n_right
            gini = (n_left * 2 * p_left * (1 - p_left)
                    + n_right * 2 * p_right * (1 - p_right)) / n

            # Coupures valides : valeurs distinctes de part et d'autre
            # et groupes assez grands (min_samples_leaf).
            valid = (xs[1:] != xs[:-1]) \
                & (n_left >= self.min_samples_leaf) \
                & (n_right >= self.min_samples_leaf)
            if not valid.any():
                continue

            gini_masked = np.where(valid, gini, np.inf)
            k = int(np.argmin(gini_masked))
            if gini_masked[k] < best_gini:
                best_gini = gini_masked[k]
                best_feature = j
                # Seuil au milieu des deux valeurs encadrant la coupure.
                best_threshold = (xs[k] + xs[k + 1]) / 2.0

        return best_feature, best_threshold

    # ---------------------------------------------------------------------------
    # Prédiction
    # ---------------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilité de défaut de chaque exemple = proportion de défauts de la
        feuille atteinte. Descente vectorisée (masques booléens propagés de nœud
        en nœud)."""
        if self.root is None:
            raise RuntimeError("Modèle non entraîné : appeler fit() d'abord.")
        X = np.asarray(X, dtype=float)
        proba = np.empty(len(X))
        self._propagate(self.root, X, np.arange(len(X)), proba)
        return proba

    def _propagate(self, node, X, idx, out):
        """Fait descendre récursivement les exemples (indices idx) dans l'arbre et
        écrit la probabilité de leur feuille dans le tableau out."""
        if node.feature is None:  # feuille
            out[idx] = node.proba
            return
        mask = X[idx, node.feature] <= node.threshold
        self._propagate(node.left, X, idx[mask], out)
        self._propagate(node.right, X, idx[~mask], out)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convertit les probabilités en classes (0/1) selon le seuil de décision."""
        return (self.predict_proba(X) >= threshold).astype(int)
