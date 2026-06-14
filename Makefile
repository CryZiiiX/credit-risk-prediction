# ======================================================================================
# Nom     : Makefile
# Rôle    : Automatise l'installation, l'exécution, les tests et le nettoyage du projet.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Exécuter make help pour lister les cibles disponibles.
# ======================================================================================

# Makefile — Pipeline Python de prédiction du risque de crédit (M1 TAA)
# L'ancien Makefile (compilation C) est conservé dans
# archive_obsolete/ancienne_version_c/Makefile.root.old

.PHONY: run compare test install clean help

help:
	@echo "Cibles disponibles :"
	@echo "  make install  - installe les dépendances Python"
	@echo "  make run      - exécute le pipeline complet (python3 main.py)"
	@echo "  make compare  - pipeline + vérification scikit-learn"
	@echo "  make test     - lance les tests (pytest)"
	@echo "  make clean    - supprime les fichiers générés par le pipeline"

install:
	pip install -r requirements.txt

run:
	python3 main.py

compare:
	python3 main.py --compare-sklearn

test:
	python3 -m pytest tests/ -v

clean:
	rm -f results/metrics.json results/predictions.csv results/sklearn_comparison.json
	rm -f results/models/*.pkl
	rm -f reports/figures/*.png
	rm -f data/splits/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
