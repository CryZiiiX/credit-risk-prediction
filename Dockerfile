# ======================================================================================
# Nom     : Dockerfile
# Rôle    : Définit l'image Docker reproductible pour exécuter le pipeline Python.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Construire avec docker build -t credit-risk . puis docker run.
# ======================================================================================

# ============================================================================
# Dockerfile - Prédiction du Risque de Crédit Bancaire (pipeline Python)
# ============================================================================
# Image Docker pour garantir la reproductibilité du projet : exécute le
# pipeline Python complet (3 modèles from scratch + métriques + figures).
#
# Construction :  docker build -t credit-risk .
# Exécution :     docker run --rm credit-risk
# ============================================================================

FROM python:3.12-slim

WORKDIR /app

# Copier requirements.txt en premier pour profiter du cache Docker
# (les dépendances changent moins souvent que le code source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet (les fichiers listés dans .dockerignore sont exclus)
COPY . .

# Vérifier le pipeline avec les tests au moment du build
RUN python3 -m pytest tests/ -q

# Commande par défaut : pipeline complet
CMD ["python3", "main.py"]
