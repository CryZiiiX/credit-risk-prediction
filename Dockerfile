# ============================================================================
# Dockerfile - Prédiction du Risque de Crédit Bancaire
# ============================================================================
# Image Docker pour garantir la reproductibilité du projet sur toutes les
# machines. Contient GCC 12, Python 3.11, et toutes les dépendances nécessaires
# pour compiler et exécuter les deux modèles ML (Régression Logistique et
# Arbre de Décision).
#
# Pour plus de détails sur les versions exactes, consultez VERSIONS.md
# ============================================================================

# Image de base officielle avec GCC 12.5.0
# Basée sur Debian 12 (Bookworm) avec Python 3.11 disponible
FROM gcc:12-bookworm

# Éviter les prompts interactifs lors de l'installation des paquets
ENV DEBIAN_FRONTEND=noninteractive

# Installer Python 3, pip et outils de build nécessaires
# Note: Python 3.11 est installé via les dépôts Debian
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    make \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier requirements.txt en premier pour optimiser le cache Docker
# (les dépendances Python changent moins souvent que le code source)
COPY python/requirements.txt .

# Installer les dépendances Python depuis requirements.txt
# Dépendances installées : pandas, numpy, matplotlib, seaborn, scikit-learn
# --break-system-packages est nécessaire avec Python 3.11+ dans Debian 12
# (acceptable dans Docker car l'environnement est isolé)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copier tout le projet dans le container
# Note: Les fichiers ignorés par .dockerignore ne seront pas copiés
COPY . .

# Compiler le projet C (inclut les deux modèles et tous les modules d'analyse)
# Le Makefile compile automatiquement tous les fichiers sources nécessaires :
# - Modèles ML : logistic_regression.c, decision_tree.c
# - Modules d'analyse : threshold_analysis.c, feature_weights_analysis.c
# - Expérimentations : learning_rate_experiment.c, iterations_experiment.c
# - Utilitaires : prepare_dataset.c
RUN make clean && make

# Commande par défaut : exécuter le programme C complet
# Le programme entraîne les deux modèles et génère toutes les métriques
CMD ["./c/build/credit_risk_predictor"]

