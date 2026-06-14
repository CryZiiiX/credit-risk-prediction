# ======================================================================================
# Nom     : src/models/__init__.py
# Rôle    : Initialise le paquet des modèles d'apprentissage implémentés from scratch.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module de paquet importé avec src.models.
# ======================================================================================

"""Modèles d'apprentissage implémentés from scratch (numpy uniquement)."""

from src.models.decision_tree import DecisionTreeClassifier
from src.models.knn import KNNClassifier
from src.models.logistic_regression import LogisticRegression

__all__ = ["LogisticRegression", "DecisionTreeClassifier", "KNNClassifier"]
