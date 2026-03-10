CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/credit_risk_predictor

# Source files
SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/utils/utils.c \
       $(SRC_DIR)/utils/memory_manager.c \
       $(SRC_DIR)/utils/csv_parser.c \
       $(SRC_DIR)/data/data_loader.c \
       $(SRC_DIR)/data/data_splitter.c \
       $(SRC_DIR)/preprocessing/preprocessing.c \
       $(SRC_DIR)/preprocessing/scaler.c \
       $(SRC_DIR)/preprocessing/encoder.c \
       $(SRC_DIR)/models/logistic_regression.c \
       $(SRC_DIR)/models/decision_tree.c \
       $(SRC_DIR)/evaluation/metrics.c \
       $(SRC_DIR)/evaluation/confusion_matrix.c \
       $(SRC_DIR)/analysis/threshold_analysis.c \
       $(SRC_DIR)/analysis/feature_weights_analysis.c

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Source files for threshold analysis
THRESHOLD_SRCS = $(SRC_DIR)/analysis/threshold_analysis.c \
                 $(SRC_DIR)/utils/utils.c \
                 $(SRC_DIR)/utils/memory_manager.c \
                 $(SRC_DIR)/utils/csv_parser.c \
                 $(SRC_DIR)/data/data_loader.c \
                 $(SRC_DIR)/data/data_splitter.c \
                 $(SRC_DIR)/preprocessing/preprocessing.c \
                 $(SRC_DIR)/preprocessing/scaler.c \
                 $(SRC_DIR)/preprocessing/encoder.c \
                 $(SRC_DIR)/models/logistic_regression.c \
                 $(SRC_DIR)/evaluation/metrics.c \
                 $(SRC_DIR)/evaluation/confusion_matrix.c

THRESHOLD_OBJS = $(THRESHOLD_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
THRESHOLD_TARGET = $(BUILD_DIR)/threshold_analysis
THRESHOLD_CFLAGS = $(CFLAGS) -DTHRESHOLD_ANALYSIS_STANDALONE

# Source files for prepare dataset
PREPARE_SRCS = $(SRC_DIR)/data/prepare_dataset.c \
               $(SRC_DIR)/utils/utils.c \
               $(SRC_DIR)/utils/memory_manager.c \
               $(SRC_DIR)/utils/csv_parser.c \
               $(SRC_DIR)/data/data_loader.c \
               $(SRC_DIR)/data/data_splitter.c \
               $(SRC_DIR)/preprocessing/preprocessing.c \
               $(SRC_DIR)/preprocessing/scaler.c \
               $(SRC_DIR)/preprocessing/encoder.c

PREPARE_OBJS = $(PREPARE_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
PREPARE_TARGET = $(BUILD_DIR)/prepare_dataset
PREPARE_CFLAGS = $(CFLAGS) -DPREPARE_DATASET_STANDALONE

# Source files for learning rate experiment
LR_EXPERIMENT_SRCS = $(SRC_DIR)/analysis/learning_rate_experiment.c \
                     $(SRC_DIR)/utils/utils.c \
                     $(SRC_DIR)/utils/memory_manager.c \
                     $(SRC_DIR)/utils/csv_parser.c \
                     $(SRC_DIR)/data/data_loader.c \
                     $(SRC_DIR)/data/data_splitter.c \
                     $(SRC_DIR)/preprocessing/preprocessing.c \
                     $(SRC_DIR)/preprocessing/scaler.c \
                     $(SRC_DIR)/preprocessing/encoder.c \
                     $(SRC_DIR)/models/logistic_regression.c \
                     $(SRC_DIR)/evaluation/metrics.c \
                     $(SRC_DIR)/evaluation/confusion_matrix.c

LR_EXPERIMENT_OBJS = $(LR_EXPERIMENT_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
LR_EXPERIMENT_TARGET = $(BUILD_DIR)/learning_rate_experiment
LR_EXPERIMENT_CFLAGS = $(CFLAGS) -DLEARNING_RATE_EXPERIMENT_STANDALONE

# Source files for iterations experiment
ITERATIONS_EXPERIMENT_SRCS = $(SRC_DIR)/analysis/iterations_experiment.c \
                             $(SRC_DIR)/utils/utils.c \
                             $(SRC_DIR)/utils/memory_manager.c \
                             $(SRC_DIR)/utils/csv_parser.c \
                             $(SRC_DIR)/data/data_loader.c \
                             $(SRC_DIR)/data/data_splitter.c \
                             $(SRC_DIR)/preprocessing/preprocessing.c \
                             $(SRC_DIR)/preprocessing/scaler.c \
                             $(SRC_DIR)/preprocessing/encoder.c \
                             $(SRC_DIR)/models/logistic_regression.c \
                             $(SRC_DIR)/evaluation/metrics.c \
                             $(SRC_DIR)/evaluation/confusion_matrix.c

ITERATIONS_EXPERIMENT_OBJS = $(ITERATIONS_EXPERIMENT_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
ITERATIONS_EXPERIMENT_TARGET = $(BUILD_DIR)/iterations_experiment
ITERATIONS_EXPERIMENT_CFLAGS = $(CFLAGS) -DITERATIONS_EXPERIMENT_STANDALONE

# Project root (parent of c/)
PROJ_ROOT = ..
# Python scripts (relative to project root, used after cd $(PROJ_ROOT))
PYTHON = python3
PLOT_SCRIPT = python/scripts/plot_results.py
COMPARE_SCRIPT = python/scripts/compare_with_sklearn.py
DETECT_OUTLIERS_SCRIPT = python/scripts/detect_outliers.py

# Results directories (relative to project root)
RESULTS_DIR = $(PROJ_ROOT)/results
METRICS_DIR = $(RESULTS_DIR)/metrics
PLOTS_DIR = $(RESULTS_DIR)/plots

.PHONY: all clean run results plots plots-all compare verify regenerate help threshold_analysis prepare_dataset learning_rate_experiment iterations_experiment outlier_values

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Build threshold analysis program
threshold_analysis: $(THRESHOLD_TARGET)

# Compilation spéciale pour threshold_analysis.c avec le flag standalone
$(BUILD_DIR)/analysis/threshold_analysis.o: $(SRC_DIR)/analysis/threshold_analysis.c
	@mkdir -p $(dir $@)
	$(CC) $(THRESHOLD_CFLAGS) -c $< -o $@

$(THRESHOLD_TARGET): $(THRESHOLD_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(THRESHOLD_CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(THRESHOLD_TARGET)"

# Build prepare dataset program
prepare_dataset: $(PREPARE_TARGET)

$(PREPARE_TARGET): $(PREPARE_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(PREPARE_CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(PREPARE_TARGET)"

# Compilation spéciale pour prepare_dataset.c avec le flag
$(BUILD_DIR)/data/prepare_dataset.o: $(SRC_DIR)/data/prepare_dataset.c
	@mkdir -p $(dir $@)
	$(CC) $(PREPARE_CFLAGS) -c $< -o $@

# Build learning rate experiment program
learning_rate_experiment: $(LR_EXPERIMENT_TARGET)

$(LR_EXPERIMENT_TARGET): $(LR_EXPERIMENT_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(LR_EXPERIMENT_CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(LR_EXPERIMENT_TARGET)"

# Compilation spéciale pour learning_rate_experiment.c avec le flag
$(BUILD_DIR)/analysis/learning_rate_experiment.o: $(SRC_DIR)/analysis/learning_rate_experiment.c
	@mkdir -p $(dir $@)
	$(CC) $(LR_EXPERIMENT_CFLAGS) -c $< -o $@

# Build iterations experiment program
iterations_experiment: $(ITERATIONS_EXPERIMENT_TARGET)

$(ITERATIONS_EXPERIMENT_TARGET): $(ITERATIONS_EXPERIMENT_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(ITERATIONS_EXPERIMENT_CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(ITERATIONS_EXPERIMENT_TARGET)"

# Compilation spéciale pour iterations_experiment.c avec le flag
$(BUILD_DIR)/analysis/iterations_experiment.o: $(SRC_DIR)/analysis/iterations_experiment.c
	@mkdir -p $(dir $@)
	$(CC) $(ITERATIONS_EXPERIMENT_CFLAGS) -c $< -o $@

clean:
	@echo "Nettoyage des fichiers de compilation..."
	@rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*/*.o $(TARGET)
	@echo "Nettoyage des résultats et graphiques..."
	@rm -rf $(RESULTS_DIR)/*
	@echo "Nettoyage des modèles sauvegardés..."
	@rm -f $(RESULTS_DIR)/models/*.bin $(RESULTS_DIR)/models/*.txt
	@echo "Clean complete"

run: $(TARGET)
	@echo "Execution du programme..."
	@cd $(PROJ_ROOT) && ./c/build/credit_risk_predictor

# Génère les résultats C (métriques, CSV, modèles)
results: $(TARGET)
	@echo "============================================================"
	@echo "ETAPE 1/4 : Generation des resultats C"
	@echo "============================================================"
	@cd $(PROJ_ROOT) && ./c/build/credit_risk_predictor
	@echo "Resultats C generes dans $(RESULTS_DIR)/"

# Génère les graphiques Python
plots: results
	@echo ""
	@echo "============================================================"
	@echo "ETAPE 2/4 : Generation des graphiques"
	@echo "============================================================"
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Erreur: $(PYTHON) n'est pas installe"; \
		exit 1; \
	fi
	@cd $(PROJ_ROOT) && $(PYTHON) $(PLOT_SCRIPT)
	@echo "Graphiques generes dans $(PLOTS_DIR)/"

# Génère tous les graphiques (C et Python) après la comparaison
plots-all: compare
	@echo ""
	@echo "============================================================"
	@echo "ETAPE 2bis/4 : Generation des graphiques (tous)"
	@echo "============================================================"
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Erreur: $(PYTHON) n'est pas installe"; \
		exit 1; \
	fi
	@cd $(PROJ_ROOT) && $(PYTHON) $(PLOT_SCRIPT)
	@echo "Graphiques generes dans $(PLOTS_DIR)/"

# Génère la comparaison scikit-learn
compare: results
	@echo ""
	@echo "============================================================"
	@echo "ETAPE 3/4 : Comparaison avec scikit-learn"
	@echo "============================================================"
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Erreur: $(PYTHON) n'est pas installe"; \
		exit 1; \
	fi
	@if ! $(PYTHON) -c "import sklearn" >/dev/null 2>&1; then \
		echo "Avertissement: scikit-learn n'est pas installe"; \
		echo "   Installation: pip install scikit-learn"; \
		exit 1; \
	fi
	@cd $(PROJ_ROOT) && $(PYTHON) $(COMPARE_SCRIPT)
	@echo "Comparaisons generees dans $(RESULTS_DIR)/"

# Détection des outliers
outlier_values:
	@echo ""
	@echo "============================================================"
	@echo "DETECTION DES OUTLIERS"
	@echo "============================================================"
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Erreur: $(PYTHON) n'est pas installe"; \
		exit 1; \
	fi
	@if ! $(PYTHON) -c "import pandas" >/dev/null 2>&1; then \
		echo "Avertissement: pandas n'est pas installe"; \
		echo "   Installation: pip install pandas"; \
		exit 1; \
	fi
	@if ! $(PYTHON) -c "import numpy" >/dev/null 2>&1; then \
		echo "Avertissement: numpy n'est pas installe"; \
		echo "   Installation: pip install numpy"; \
		exit 1; \
	fi
	@mkdir -p $(PROJ_ROOT)/data/stats
	@cd $(PROJ_ROOT) && $(PYTHON) $(DETECT_OUTLIERS_SCRIPT)
	@echo "Analyse des outliers sauvegardee dans data/stats/outliers_analysis.txt"

# Vérifie que tous les fichiers sont générés
verify:
	@echo ""
	@echo "============================================================"
	@echo "ETAPE 4/4 : Verification des resultats"
	@echo "============================================================"
	@echo "Fichiers de metriques:"
	@echo "   Regression Logistique: $$(ls -1 $(METRICS_DIR)/logistic_regression/*.txt 2>/dev/null | wc -l | xargs)"; \
	echo "   Arbre de Decision: $$(ls -1 $(METRICS_DIR)/decision_tree/*.txt 2>/dev/null | wc -l | xargs)"; \
	echo "   Total: $$(find $(METRICS_DIR)/logistic_regression $(METRICS_DIR)/decision_tree -name '*.txt' 2>/dev/null | wc -l | xargs)"
	@echo "Graphiques PNG:"
	@ls -1 $(PLOTS_DIR)/*.png 2>/dev/null | wc -l | xargs echo "   -"
	@echo "Fichiers de comparaison:"
	@ls -1 $(RESULTS_DIR)/sklearn*.txt 2>/dev/null | wc -l | xargs echo "   -"
	@echo ""
	@echo "Resume des metriques principales:"
	@if [ -f $(METRICS_DIR)/logistic_regression/lr_c_test_metrics.txt ]; then \
		echo "   Regression Logistique:"; \
		grep "Accuracy:" $(METRICS_DIR)/logistic_regression/lr_c_test_metrics.txt | head -1 | sed 's/^/     /'; \
	fi
	@if [ -f $(METRICS_DIR)/decision_tree/dt_c_test_metrics.txt ]; then \
		echo "   Arbre de Decision:"; \
		grep "Accuracy:" $(METRICS_DIR)/decision_tree/dt_c_test_metrics.txt | head -1 | sed 's/^/     /'; \
	fi
	@echo ""
	@echo "Fichiers de statistiques arbre de décision:"
	@if [ -f $(METRICS_DIR)/decision_tree/dt_c_tree_stats.txt ]; then \
		echo "   [OK] dt_c_tree_stats.txt"; \
		grep "Profondeur réelle:" $(METRICS_DIR)/decision_tree/dt_c_tree_stats.txt | sed 's/^/     /'; \
		grep "Nombre total de nœuds:" $(METRICS_DIR)/decision_tree/dt_c_tree_stats.txt | sed 's/^/     /'; \
		grep "Temps d'entraînement:" $(METRICS_DIR)/decision_tree/dt_c_tree_stats.txt | sed 's/^/     /'; \
	else \
		echo "   [MANQUANT] dt_c_tree_stats.txt"; \
	fi
	@if [ -f $(METRICS_DIR)/decision_tree/dt_python_tree_stats.txt ]; then \
		echo "   [OK] dt_python_tree_stats.txt"; \
		grep "Profondeur réelle:" $(METRICS_DIR)/decision_tree/dt_python_tree_stats.txt | sed 's/^/     /'; \
		grep "Nombre total de nœuds:" $(METRICS_DIR)/decision_tree/dt_python_tree_stats.txt | sed 's/^/     /'; \
		grep "Temps d'entraînement:" $(METRICS_DIR)/decision_tree/dt_python_tree_stats.txt | sed 's/^/     /'; \
	else \
		echo "   [MANQUANT] dt_python_tree_stats.txt"; \
	fi
	@echo ""
	@echo "Verification terminee"

# Régénère tout : compile, exécute, génère graphiques et compare
regenerate: clean all results compare plots-all verify
	@echo ""
	@echo "============================================================"
	@echo "REGENERATION COMPLETE TERMINEE"
	@echo "============================================================"
	@echo "Tous les résultats ont été régénérés avec succès !"
	@echo ""

# Aide
help:
	@echo "Makefile - Prédiction du Risque de Crédit"
	@echo ""
	@echo "Cibles disponibles:"
	@echo "  make              - Compile le projet"
	@echo "  make clean         - Nettoie les fichiers de compilation"
	@echo "  make run           - Compile et exécute le programme"
	@echo "  make results       - Génère les résultats C (métriques, CSV, modèles)"
	@echo "  make plots         - Génère les graphiques Python (nécessite results)"
	@echo "  make plots-all     - Génère tous les graphiques (C et Python, nécessite compare)"
	@echo "  make compare       - Génère la comparaison avec scikit-learn"
	@echo "  make verify        - Vérifie que tous les fichiers sont générés"
	@echo "  make regenerate    - Régénère tout (clean + compile + results + compare + plots-all + verify)"
	@echo "  make help          - Affiche cette aide"
	@echo ""
