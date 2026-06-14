# ======================================================================================
# Nom     : src/models/logistic_regression.py
# Rôle    : Implémente le modèle de régression logistique entraîné par descente de gradient.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé pour entraîner et évaluer le modèle associé.
# ======================================================================================

"""Régression logistique binaire, entraînée par descente de gradient.

Modèle : p(y=1|x) = sigmoïde(w·x + b)
Coût : entropie croisée (log-loss), convexe → la descente de gradient
converge vers le minimum global si le taux d'apprentissage est raisonnable.

Hyperparamètres hérités de la version 1 du projet (validés expérimentalement) :
learning_rate=0.1, n_iterations=1000, sur données normalisées.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoïde : transforme un score réel en probabilité (entre 0 et 1).
    Deux formules selon le signe de z, pour éviter un dépassement numérique."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    """Régression logistique from scratch (descente de gradient full-batch)."""

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.cost_history = []  # suivi de la convergence (pour le rapport)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Apprend les poids et le biais par descente de gradient (full batch).

        À chaque itération : prédiction p, erreur (p - y), puis mise à jour des
        poids dans la direction qui réduit la log-loss (dw = X^T (p - y) / n).
        Le coût est suivi pour vérifier la convergence.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            p = sigmoid(X @ self.weights + self.bias)
            error = p - y

            self.weights -= self.learning_rate * (X.T @ error) / n_samples
            self.bias -= self.learning_rate * error.mean()

            # Log-loss avec clipping pour éviter log(0).
            p_clip = np.clip(p, 1e-12, 1 - 1e-12)
            cost = -np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
            self.cost_history.append(float(cost))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Renvoie la probabilité estimée de défaut pour chaque exemple."""
        if self.weights is None:
            raise RuntimeError("Modèle non entraîné : appeler fit() d'abord.")
        return sigmoid(np.asarray(X, dtype=float) @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convertit les probabilités en classes (0/1) selon le seuil de décision."""
        return (self.predict_proba(X) >= threshold).astype(int)
