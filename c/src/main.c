/*****************************************************************************************************

Nom : src/main.c

Rôle : Fonction main() orchestrant le pipeline complet (chargement, prétraitement, entraînement, évaluation)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data/data_loader.h"
#include "data/data_splitter.h"
#include "preprocessing/preprocessing.h"
#include "preprocessing/scaler.h"
#include "models/logistic_regression.h"
#include "models/decision_tree.h"
#include "evaluation/metrics.h"
#include "evaluation/confusion_matrix.h"
#include "analysis/threshold_analysis.h"
#include "analysis/feature_weights_analysis.h"

/**
 * Fonction : save_decision_tree_stats
 * Rôle     : Sauvegarde les statistiques de l'arbre de décision (profondeur, nœuds, temps)
 * Param    : filename (nom du fichier de destination), tree (arbre de décision), training_time (temps d'entraînement en secondes)
 * Retour   : void
 */
void save_decision_tree_stats(const char* filename, DecisionTree* tree, double training_time) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "============================================================\n");
    fprintf(file, "STATISTIQUES ARBRE DE DÉCISION - IMPLEMENTATION C\n");
    fprintf(file, "============================================================\n\n");
    fprintf(file, "Profondeur réelle: %d\n", get_tree_depth(tree));
    fprintf(file, "Nombre total de nœuds: %d\n", count_tree_nodes(tree));
    fprintf(file, "Temps d'entraînement: environ %.3f secondes\n", training_time);
    fprintf(file, "\n");
    fprintf(file, "Hyperparamètres:\n");
    fprintf(file, "  - max_depth: %d\n", tree->max_depth);
    fprintf(file, "  - min_samples_split: %d\n", tree->min_samples_split);
    fprintf(file, "  - min_samples_leaf: %d\n", tree->min_samples_leaf);
    fprintf(file, "  - criterion: %s\n", tree->criterion == GINI ? "Gini" : "Entropy");
    
    fclose(file);
}

/**
 * Fonction : save_performance_benchmark
 * Rôle     : Sauvegarde les résultats de benchmarking dans un fichier
 * Param    : time_loading (temps de chargement), time_preprocessing (temps de prétraitement), time_normalization (temps de normalisation), time_training (temps d'entraînement), time_evaluation (temps d'évaluation)
 * Retour   : void
 */
void save_performance_benchmark(double time_loading, double time_preprocessing, 
                                double time_normalization, double time_training, 
                                double time_evaluation) {
    // Créer le répertoire si nécessaire
    int ret = system("mkdir -p results/metrics/logistic_regression");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to create directory\n");
    }
    
    FILE* file = fopen("results/metrics/logistic_regression/lr_c_performance_benchmark.txt", "w");
    if (!file) {
        fprintf(stderr, "Erreur: Impossible de créer le fichier de benchmarking\n");
        return;
    }
    
    // Calculer le temps total
    double total_time = time_loading + time_preprocessing + time_normalization + 
                       time_training + time_evaluation;
    
    // En-tête
    fprintf(file, "================================================================================\n");
    fprintf(file, "PERFORMANCE COMPUTATIONNELLE - REGRESSION LOGISTIQUE\n");
    fprintf(file, "================================================================================\n\n");
    
    // En-tête du tableau
    fprintf(file, "%-45s %-20s %-25s\n", "Opération", "Temps (secondes)", "Pourcentage du total");
    fprintf(file, "--------------------------------------------------------------------------------\n");
    
    // Données
    double pct_loading = total_time > 0 ? (time_loading / total_time * 100.0) : 0.0;
    double pct_preprocessing = total_time > 0 ? (time_preprocessing / total_time * 100.0) : 0.0;
    double pct_normalization = total_time > 0 ? (time_normalization / total_time * 100.0) : 0.0;
    double pct_training = total_time > 0 ? (time_training / total_time * 100.0) : 0.0;
    double pct_evaluation = total_time > 0 ? (time_evaluation / total_time * 100.0) : 0.0;
    
    fprintf(file, "%-45s %-20.3f %-25.1f%%\n", 
            "Chargement CSV + encodage", time_loading, pct_loading);
    
    fprintf(file, "%-45s %-20.3f %-25.1f%%\n", 
            "Prétraitement (imputation + shuffle)", time_preprocessing, pct_preprocessing);
    
    fprintf(file, "%-45s %-20.3f %-25.1f%%\n", 
            "Normalisation (fit + transform)", time_normalization, pct_normalization);
    
    fprintf(file, "%-45s %-20.3f %-25.1f%%\n", 
            "Entraînement (1000 itérations)", time_training, pct_training);
    
    fprintf(file, "%-45s %-20.3f %-25.1f%%\n", 
            "Évaluation (prédictions + métriques)", time_evaluation, pct_evaluation);
    
    fprintf(file, "--------------------------------------------------------------------------------\n");
    fprintf(file, "%-45s %-20.3f %-25.1f%%\n", "TOTAL", total_time, 100.0);
    
    fclose(file);
    printf("\n[OK] Performance benchmark saved: results/metrics/logistic_regression/lr_c_performance_benchmark.txt\n");
}

/**
 * Fonction : main
 * Rôle     : Orchestre le pipeline complet de machine learning (chargement, prétraitement, entraînement, évaluation)
 * Param    : argc (nombre d'arguments), argv (tableau d'arguments)
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    printf("Credit Risk Prediction - Machine Learning in C\n");
    printf("===============================================\n\n");
    
    // Variables pour mesurer les temps
    double time_loading = 0.0;
    double time_preprocessing = 0.0;
    double time_normalization = 0.0;
    double time_training = 0.0;
    double time_evaluation = 0.0;
    clock_t start, end;
    
    /* **************************************************
     * # --- CHARGEMENT OU GÉNÉRATION DU SPLIT --- #
     * ************************************************** */
    
    SplitData* split = NULL;
    Dataset* dataset = NULL;
    
    // Essayer de charger les datasets prétraités
    printf("Vérification des datasets prétraités...\n");
    start = clock();
    // Utiliser load_csv_numeric pour charger les datasets prétraités (déjà encodés)
    Dataset* train = load_csv_numeric("data/processed/train.csv", 1, 8);
    Dataset* test = load_csv_numeric("data/processed/test.csv", 1, 8);
    end = clock();
    
    if (train && test && train->rows > 0 && test->rows > 0) {
        // Datasets existants trouvés
        time_loading = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("[OK] Datasets prétraités trouvés!\n");
        printf("Train set: %d échantillons\n", train->rows);
        printf("Test set: %d échantillons\n\n", test->rows);
        
        // Créer une structure SplitData
        split = (SplitData*)malloc(sizeof(SplitData));
        if (!split) {
            fprintf(stderr, "Erreur: Allocation mémoire échouée\n");
            free_dataset(train);
            free_dataset(test);
            return 1;
        }
        split->train = train;
        split->test = test;
    } else {
        // Fichiers non trouvés ou vides, faire le split comme avant
        printf("[INFO] Datasets prétraités non trouvés. Génération du split...\n\n");
        
        // Mesurer le chargement CSV + encodage
        printf("Loading dataset...\n");
        start = clock();
        dataset = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);
        if (!dataset) {
            fprintf(stderr, "Error loading dataset\n");
            if (train) free_dataset(train);
            if (test) free_dataset(test);
            return 1;
        }
        end = clock();
        time_loading = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Dataset loaded: %d samples, %d features\n", dataset->rows, dataset->cols);
        
        // Afficher statistiques des classes
        int class_0 = 0, class_1 = 0;
        for (int i = 0; i < dataset->rows; i++) {
            if (dataset->labels[i] == 0) class_0++;
            else class_1++;
        }
        printf("Balance des classes: Classe 0 (pas de défaut) = %d (%.1f%%), Classe 1 (défaut) = %d (%.1f%%)\n\n",
               class_0, 100.0 * class_0 / dataset->rows,
               class_1, 100.0 * class_1 / dataset->rows);
        
        /* **************************************************
         * # --- PRÉTRAITEMENT --- #
         * ************************************************** */
        
        // Mesurer le prétraitement (imputation + shuffle)
        printf("Preprocessing data...\n");
        start = clock();
        preprocess_dataset(dataset);
        srand(42);  // Pour reproductibilité
        shuffle_dataset(dataset);
        end = clock();
        time_preprocessing = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        /* **************************************************
         * # --- DIVISION TRAIN/TEST --- #
         * ************************************************** */
        
        printf("Splitting dataset (80%% train, 20%% test)...\n");
        split = split_dataset(dataset, 0.8);
        printf("Train set: %d samples\n", split->train->rows);
        printf("Test set: %d samples\n\n", split->test->rows);
        
        /* **************************************************
         * # --- SAUVEGARDE DES DATASETS --- #
         * ************************************************** */
        
        printf("Saving train and test sets...\n");
        int ret_save = system("mkdir -p data/processed");
        if (ret_save != 0) {
            fprintf(stderr, "Warning: Failed to create directory data/processed\n");
        }
        
        // Sauvegarder les datasets (après prétraitement, avant normalisation)
        save_dataset_with_header("data/processed/train.csv", split->train);
        save_dataset_with_header("data/processed/test.csv", split->test);
        printf("[OK] Train set saved: data/processed/train.csv\n");
        printf("[OK] Test set saved: data/processed/test.csv\n\n");
    }
    
    /* **************************************************
     * # --- NORMALISATION --- #
     * ************************************************** */
    
    // Mesurer la normalisation
    Scaler* scaler = NULL;
    
    // Essayer de charger le scaler existant
    printf("Vérification du scaler...\n");
    start = clock();
    scaler = load_scaler("data/processed/scaler_params.txt");
    
    if (scaler) {
        printf("[OK] Scaler chargé depuis data/processed/scaler_params.txt\n");
        transform_dataset(split->train, scaler);
        transform_dataset(split->test, scaler);
    } else {
        // Scaler non trouvé, le calculer
        printf("[INFO] Scaler non trouvé. Calcul du scaler sur le train set...\n");
        scaler = fit_scaler(split->train);
        transform_dataset(split->train, scaler);
        transform_dataset(split->test, scaler);
        save_scaler("data/processed/scaler_params.txt", scaler);
        printf("[OK] Scaler sauvegardé: data/processed/scaler_params.txt\n");
    }
    end = clock();
    time_normalization = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("\n");
    
    /* **************************************************
     * # --- ENTRAÎNEMENT RÉGRESSION LOGISTIQUE --- #
     * ************************************************** */
    
    // Mesurer l'entraînement
    printf("\nTraining Logistic Regression...\n");
    LogisticRegression* model = create_logistic_regression(
        split->train->cols, 
        0.01,    // learning rate
        1000     // max iterations
    );
    start = clock();
    train_logistic_regression(model, split->train);
    end = clock();
    time_training = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Générer le tableau de convergence
    printf("\nGénération du tableau de convergence...\n");
    generate_convergence_table(
        "results/plots/csv/lr_c_cost_curve.csv",
        "results/metrics/logistic_regression/lr_c_convergence_table.txt"
    );
    
    // Save model
    {
        int ret_mkdir = system("mkdir -p results/models");
        if (ret_mkdir != 0) {
            fprintf(stderr, "Warning: Failed to create directory results/models\n");
        }
    }
    save_model("results/models/logistic_model.bin", model);
    printf("\nModel saved to results/models/logistic_model.bin\n");
    
    /* **************************************************
     * # --- ÉVALUATION RÉGRESSION LOGISTIQUE --- #
     * ************************************************** */
    
    printf("\n--- Training Set Evaluation ---\n");
    int* train_predictions = predict(model, split->train);
    print_metrics(split->train->labels, train_predictions, split->train->rows);
    int ret = system("mkdir -p results/metrics/logistic_regression");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to create directory results/metrics/logistic_regression\n");
    }
    save_metrics("results/metrics/logistic_regression/lr_c_train_metrics.txt", split->train->labels, train_predictions, split->train->rows);
    
    ConfusionMatrix* train_cm = compute_confusion_matrix(split->train->labels, train_predictions, split->train->rows);
    print_confusion_matrix(train_cm);
    
    // Evaluate on test set
    // Mesurer l'évaluation (prédictions + métriques)
    printf("\n--- Test Set Evaluation ---\n");
    start = clock();
    int* test_predictions = predict(model, split->test);
    print_metrics(split->test->labels, test_predictions, split->test->rows);
    
    // Compute AUC-ROC and save ROC curve data
    double* test_probabilities = predict_proba(model, split->test);
    ret = system("mkdir -p results/plots/csv");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to create directory results/plots/csv\n");
    }
    double lr_auc_roc = compute_auc_roc_and_save(test_probabilities, split->test->labels, split->test->rows, "results/plots/csv/lr_roc_data.csv");
    printf("AUC-ROC:   %.4f\n", lr_auc_roc);
    
    save_metrics_with_auc_and_title("results/metrics/logistic_regression/lr_c_test_metrics.txt", "REGRESSION LOGISTIQUE - IMPLEMENTATION C", split->test->labels, test_predictions, split->test->rows, lr_auc_roc);
    
    ConfusionMatrix* test_cm = compute_confusion_matrix(split->test->labels, test_predictions, split->test->rows);
    print_confusion_matrix(test_cm);
    save_confusion_matrix("results/metrics/logistic_regression/lr_c_test_confusion_matrix.txt", test_cm);
    end = clock();
    time_evaluation = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Générer l'analyse des seuils optimaux
    printf("\nGenerating threshold analysis...\n");
    generate_threshold_analysis_table(model, split->test);
    printf("[OK] Threshold analysis saved: results/metrics/logistic_regression/lr_c_threshold_analysis.txt\n");
    
    // Générer l'analyse des poids des features
    printf("\nGenerating feature weights analysis...\n");
    generate_feature_weights_table(model);
    printf("[OK] Feature weights analysis saved: results/metrics/logistic_regression/lr_c_feature_weights_analysis.txt\n");
    
    // Sauvegarder le benchmark de performance
    save_performance_benchmark(time_loading, time_preprocessing, time_normalization, 
                              time_training, time_evaluation);
    
    double lr_train_acc = compute_accuracy(split->train->labels, train_predictions, split->train->rows);
    double lr_test_acc = compute_accuracy(split->test->labels, test_predictions, split->test->rows);
    double lr_test_f1 = compute_f1_score(split->test->labels, test_predictions, split->test->rows);
    
    /* **************************************************
     * # --- ENTRAÎNEMENT ARBRE DE DÉCISION --- #
     * ************************************************** */
    
    printf("\n\n=== DECISION TREE ===\n");
    printf("Training Decision Tree (max_depth=7, min_samples_split=20, min_samples_leaf=10, criterion=GINI)...\n");
    DecisionTree* dt = create_decision_tree(7, 20, 10, GINI);
    
    // Mesurer le temps d'entraînement
    start = clock();
    train_decision_tree(dt, split->train);
    end = clock();
    double training_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Decision Tree trained successfully!\n");
    printf("Actual tree depth: %d\n", get_tree_depth(dt));
    printf("Total nodes: %d\n", count_tree_nodes(dt));
    printf("Training time: %.3f seconds\n\n", training_time);
    
    // Créer le répertoire pour l'arbre de décision
    ret = system("mkdir -p results/metrics/decision_tree");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to create directory results/metrics/decision_tree\n");
    }
    // Sauvegarder les statistiques de l'arbre
    save_decision_tree_stats("results/metrics/decision_tree/dt_c_tree_stats.txt", dt, training_time);
    
    /* **************************************************
     * # --- ÉVALUATION ARBRE DE DÉCISION --- #
     * ************************************************** */
    
    printf("--- Decision Tree: Training Set ---\n");
    int* dt_train_pred = predict_tree_dataset(dt, split->train);
    print_metrics(split->train->labels, dt_train_pred, split->train->rows);
    save_metrics("results/metrics/decision_tree/dt_c_train_metrics.txt", split->train->labels, dt_train_pred, split->train->rows);
    
    // Evaluate Decision Tree on test set
    printf("\n--- Decision Tree: Test Set ---\n");
    int* dt_test_pred = predict_tree_dataset(dt, split->test);
    print_metrics(split->test->labels, dt_test_pred, split->test->rows);
    
    double* dt_test_proba = get_tree_probabilities(dt, split->test);
    double dt_auc_roc = compute_auc_roc_and_save(dt_test_proba, split->test->labels, split->test->rows, "results/plots/csv/dt_roc_data.csv");
    printf("AUC-ROC:   %.4f\n", dt_auc_roc);
    
    save_metrics_with_auc_and_title("results/metrics/decision_tree/dt_c_test_metrics.txt", "ARBRE DE DECISION - IMPLEMENTATION C", split->test->labels, dt_test_pred, split->test->rows, dt_auc_roc);
    
    ConfusionMatrix* dt_test_cm = compute_confusion_matrix(split->test->labels, dt_test_pred, split->test->rows);
    print_confusion_matrix(dt_test_cm);
    save_confusion_matrix("results/metrics/decision_tree/dt_c_test_confusion_matrix.txt", dt_test_cm);
    
    save_decision_tree("results/models/decision_tree_model.bin", dt);
    
    double dt_train_acc = compute_accuracy(split->train->labels, dt_train_pred, split->train->rows);
    double dt_test_acc = compute_accuracy(split->test->labels, dt_test_pred, split->test->rows);
    double dt_test_f1 = compute_f1_score(split->test->labels, dt_test_pred, split->test->rows);
    
    /* **************************************************
     * # --- COMPARAISON DES MODÈLES --- #
     * ************************************************** */
    
    printf("\n\n=== MODEL COMPARISON ===\n");
    printf("+-----------------------+----------+----------+----------+\n");
    printf("| Model                 | Accuracy | F1-Score | AUC-ROC  |\n");
    printf("+-----------------------+----------+----------+----------+\n");
    printf("| Logistic Regression   | %.4f   | %.4f   | %.4f   |\n", lr_test_acc, lr_test_f1, lr_auc_roc);
    printf("| Decision Tree         | %.4f   | %.4f   | %.4f   |\n", dt_test_acc, dt_test_f1, dt_auc_roc);
    printf("+-----------------------+----------+----------+----------+\n");
    
    printf("\nTrain vs Test Accuracy:\n");
    printf("Logistic Regression: Train=%.4f, Test=%.4f, Gap=%.4f\n", 
           lr_train_acc, lr_test_acc, lr_train_acc - lr_test_acc);
    printf("Decision Tree:       Train=%.4f, Test=%.4f, Gap=%.4f\n", 
           dt_train_acc, dt_test_acc, dt_train_acc - dt_test_acc);
    
    printf("\n\nResults saved in results/ directory\n");
    
    /* **************************************************
     * # --- NETTOYAGE MÉMOIRE --- #
     * ************************************************** */
    
    free(train_predictions);
    free(test_predictions);
    free(test_probabilities);
    free(dt_train_pred);
    free(dt_test_pred);
    free(dt_test_proba);
    free_confusion_matrix(train_cm);
    free_confusion_matrix(test_cm);
    free_confusion_matrix(dt_test_cm);
    free_decision_tree(dt);
    free_logistic_regression(model);
    free_scaler(scaler);
    free_split_data(split);
    // dataset peut être NULL si les fichiers ont été chargés directement
    if (dataset) {
        free_dataset(dataset);
    }
    
    return 0;
}
