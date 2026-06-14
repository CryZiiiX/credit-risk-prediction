# ======================================================================================
# Nom     : src/models/knn.py
# Rôle    : Implémente la méthode des k plus proches voisins (k-NN).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé pour entraîner et évaluer le modèle associé.
# ======================================================================================

"""k plus proches voisins (k-NN), from scratch.

Méthode non paramétrique : il n'y a pas de phase d'apprentissage, le
« modèle » est l'ensemble d'entraînement lui-même. Pour prédire un exemple,
on calcule sa distance euclidienne à tous les exemples du train et on vote
parmi les k plus proches.

Deux points d'implémentation importants :
- la NORMALISATION des données est indispensable (sinon une variable à
  grande échelle comme le revenu écrase toutes les autres dans la distance) ;
- la matrice de distances complète (n_test × n_train) serait trop grosse en
  mémoire (~900 Mo) : on traite donc les exemples de test par paquets
  (chunks) de quelques centaines de lignes.
"""

import numpy as np


class KNNClassifier:
    """k-NN binaire avec vote majoritaire (probabilité = moyenne des labels
    des k voisins)."""

    def __init__(self, k: int = 11, chunk_size: int = 512):
        self.k = k
        self.chunk_size = chunk_size
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Mémorise les données d'entraînement (le k-NN n'a pas d'apprentissage)."""
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        return self

    def _neighbor_labels(self, X: np.ndarray, k: int) -> np.ndarray:
        """Retrouve les classes des k plus proches voisins de chaque exemple.

        Travail par paquets (la matrice complète des distances serait trop grosse),
        distance via ||a-b||² = ||a||² + ||b||² - 2 a·b (le terme ||a||² est ignoré
        car constant), et np.argpartition pour les k plus proches sans tout trier.
        """
        if self.X_train is None:
            raise RuntimeError("Modèle non entraîné : appeler fit() d'abord.")
        X = np.asarray(X, dtype=float)
        train_sq = np.sum(self.X_train ** 2, axis=1)

        labels = np.empty((len(X), k))
        for start in range(0, len(X), self.chunk_size):
            chunk = X[start:start + self.chunk_size]
            # Distances au carré (à une constante près) : n_chunk × n_train
            d2 = train_sq - 2 * chunk @ self.X_train.T
            # argpartition : trouve les k plus petits sans trier le reste
            # (O(n) au lieu de O(n log n)).
            nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
            labels[start:start + len(chunk)] = self.y_train[nearest]
        return labels

    def predict_proba(self, X: np.ndarray, k: int = None) -> np.ndarray:
        """Probabilité de défaut = proportion de défauts parmi les k voisins.
        Un k différent peut être passé pour tester plusieurs valeurs sans refaire
        le fit."""
        k = self.k if k is None else k
        return self._neighbor_labels(X, k).mean(axis=1)

    def predict(self, X: np.ndarray, threshold: float = 0.5,
                k: int = None) -> np.ndarray:
        """Convertit les probabilités en classes (0/1) selon le seuil de décision."""
        return (self.predict_proba(X, k=k) >= threshold).astype(int)
