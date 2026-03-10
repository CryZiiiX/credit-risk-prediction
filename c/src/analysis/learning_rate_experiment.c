/*****************************************************************************************************

Nom : src/analysis/learning_rate_experiment.c

Rôle : Programme standalone pour tester différentes valeurs de learning rate

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make learning_rate_experiment

    Pour executer : ./build/learning_rate_experiment

******************************************************************************************************/

#include "learning_rate_experiment.h"
#include "../models/logistic_regression.h"
#include "../data/data_loader.h"
#include "../data/data_splitter.h"
#include "../preprocessing/preprocessing.h"
#include "../preprocessing/scaler.h"
#include "../evaluation/metrics.h"
#include "../utils/csv_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* **************************************************
 * # --- CONSTANTES --- #
 * ************************************************** */

#define NUM_LEARNING_RATES 6
#define MAX_ITERATIONS 1000

// Valeurs de learning rate à tester
static const double LEARNING_RATES[NUM_LEARNING_RATES] = {0.001, 0.005, 0.01, 0.05, 0.1, 0.5};

/* **************************************************
 * # --- FONCTIONS UTILITAIRES --- #
 * ************************************************** */

/**
 * Fonction : determine_convergence_status
 * Rôle     : Détermine l'état de convergence basé sur le coût final
 * Param    : final_cost (coût final), cost_history (historique des coûts, peut être NULL)
 * Retour   : void (remplit le buffer status)
 */
void determine_convergence_status(double final_cost, char* status, size_t status_size) {
    // Vérifier si le coût est invalide (NaN ou Inf)
    if (!isfinite(final_cost) || final_cost < 0) {
        snprintf(status, status_size, "Divergence");
        return;
    }
    
    // Déterminer l'état selon le coût final
    if (final_cost < 0.45) {
        snprintf(status, status_size, "Oui (coût = %.2f)", final_cost);
    } else if (final_cost < 0.50) {
        snprintf(status, status_size, "Partielle (coût = %.2f)", final_cost);
    } else if (final_cost < 1.0) {
        snprintf(status, status_size, "Non (coût = %.2f)", final_cost);
    } else {
        snprintf(status, status_size, "Divergence");
    }
}

/**
 * Fonction : load_datasets
 * Rôle     : Charge les datasets train et test (ou les régénère si absents)
 * Param    : train (pointeur vers Dataset* pour train), test (pointeur vers Dataset* pour test), scaler (pointeur vers Scaler*)
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int load_datasets(Dataset** train, Dataset** test, Scaler** scaler) {
    // Essayer de charger les datasets prétraités
    *train = load_csv_numeric("data/processed/train.csv", 1, 8);
    *test = load_csv_numeric("data/processed/test.csv", 1, 8);
    
    if (*train && *test && (*train)->rows > 0 && (*test)->rows > 0) {
        printf("[OK] Datasets chargés depuis data/processed/\n");
        printf("Train: %d échantillons, Test: %d échantillons\n", (*train)->rows, (*test)->rows);
        
        // Charger le scaler
        *scaler = load_scaler("data/processed/scaler_params.txt");
        if (*scaler) {
            printf("[OK] Scaler chargé\n");
            transform_dataset(*train, *scaler);
            transform_dataset(*test, *scaler);
        } else {
            printf("[INFO] Scaler non trouvé, calcul en cours...\n");
            *scaler = fit_scaler(*train);
            transform_dataset(*train, *scaler);
            transform_dataset(*test, *scaler);
        }
        return 0;
    }
    
    // Si les fichiers n'existent pas, régénérer
    printf("[INFO] Datasets non trouvés, régénération...\n");
    
    Dataset* dataset = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);
    if (!dataset) {
        fprintf(stderr, "Erreur: Impossible de charger le dataset\n");
        return 1;
    }
    
    preprocess_dataset(dataset);
    srand(42);
    shuffle_dataset(dataset);
    SplitData* split = split_dataset(dataset, 0.8);
    
    *train = split->train;
    *test = split->test;
    
    *scaler = fit_scaler(*train);
    transform_dataset(*train, *scaler);
    transform_dataset(*test, *scaler);
    
    free_dataset(dataset);
    free(split);
    
    printf("[OK] Datasets régénérés\n");
    return 0;
}

/**
 * Fonction : save_experiment_results
 * Rôle     : Sauvegarde les résultats de l'expérimentation dans un fichier
 * Param    : results (tableau de résultats), n_results (nombre de résultats)
 * Retour   : void
 */
void save_experiment_results(LearningRateResult* results, int n_results) {
    // Créer le répertoire si nécessaire
    int ret = system("mkdir -p results/metrics/logistic_regression");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to create directory\n");
    }
    
    FILE* file = fopen("results/metrics/logistic_regression/lr_c_learning_rate_experiment.txt", "w");
    if (!file) {
        fprintf(stderr, "Erreur: Impossible de créer le fichier de résultats\n");
        return;
    }
    
    // En-tête
    fprintf(file, "================================================================================\n");
    fprintf(file, "EXPÉRIENCE 1 : VARIATION DU LEARNING RATE\n");
    fprintf(file, "================================================================================\n\n");
    
    // En-tête du tableau
    fprintf(file, "%-18s %-12s %-30s %-15s %-10s %-10s\n",
            "Learning rate (α)", "Iterations", "Convergence", "Accuracy Test", "F1-Score", "Temps (s)");
    fprintf(file, "--------------------------------------------------------------------------------\n");
    
    // Données
    for (int i = 0; i < n_results; i++) {
        fprintf(file, "%-18.3f %-12d %-30s %-15.2f%% %-10.2f%% %-10.2f\n",
                results[i].learning_rate,
                results[i].iterations,
                results[i].convergence_status,
                results[i].test_accuracy * 100.0,
                results[i].test_f1 * 100.0,
                results[i].training_time);
    }
    
    fprintf(file, "--------------------------------------------------------------------------------\n");
    
    fclose(file);
    printf("\n[OK] Résultats sauvegardés: results/metrics/logistic_regression/lr_c_learning_rate_experiment.txt\n");
}

/**
 * Fonction : run_learning_rate_experiment
 * Rôle     : Lance l'expérimentation complète avec différents learning rates
 * Param    : aucun
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int run_learning_rate_experiment(void) {
    printf("================================================================================\n");
    printf("EXPÉRIENCE 1 : VARIATION DU LEARNING RATE\n");
    printf("================================================================================\n\n");
    
    // Charger les datasets
    Dataset* train = NULL;
    Dataset* test = NULL;
    Scaler* scaler = NULL;
    
    if (load_datasets(&train, &test, &scaler) != 0) {
        fprintf(stderr, "Erreur: Impossible de charger les datasets\n");
        return 1;
    }
    
    if (!train || !test || train->cols != test->cols) {
        fprintf(stderr, "Erreur: Datasets invalides\n");
        if (train) free_dataset(train);
        if (test) free_dataset(test);
        if (scaler) free_scaler(scaler);
        return 1;
    }
    
    int n_features = train->cols;
    LearningRateResult* results = (LearningRateResult*)malloc(NUM_LEARNING_RATES * sizeof(LearningRateResult));
    if (!results) {
        fprintf(stderr, "Erreur: Allocation mémoire échouée\n");
        free_dataset(train);
        free_dataset(test);
        free_scaler(scaler);
        return 1;
    }
    
    printf("\nDémarrage des expérimentations...\n");
    printf("Nombre de learning rates à tester: %d\n\n", NUM_LEARNING_RATES);
    
    // Boucle d'expérimentation
    for (int i = 0; i < NUM_LEARNING_RATES; i++) {
        double lr = LEARNING_RATES[i];
        printf("--- Test avec learning rate = %.3f ---\n", lr);
        
        // Créer le modèle
        LogisticRegression* model = create_logistic_regression(n_features, lr, MAX_ITERATIONS);
        if (!model) {
            fprintf(stderr, "Erreur: Impossible de créer le modèle\n");
            continue;
        }
        
        // Mesurer le temps d'entraînement
        clock_t start = clock();
        double final_cost = train_logistic_regression_with_cost(model, train);
        clock_t end = clock();
        double training_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Déterminer l'état de convergence
        determine_convergence_status(final_cost, results[i].convergence_status, 
                                    sizeof(results[i].convergence_status));
        
        // Évaluer sur le test set
        int* predictions = predict(model, test);
        double accuracy = compute_accuracy(test->labels, predictions, test->rows);
        double f1 = compute_f1_score(test->labels, predictions, test->rows);
        
        // Stocker les résultats
        results[i].learning_rate = lr;
        results[i].iterations = MAX_ITERATIONS;
        results[i].final_cost = final_cost;
        results[i].test_accuracy = accuracy;
        results[i].test_f1 = f1;
        results[i].training_time = training_time;
        
        // Afficher les résultats
        printf("  Coût final: %.4f\n", final_cost);
        printf("  Convergence: %s\n", results[i].convergence_status);
        printf("  Accuracy: %.2f%%\n", accuracy * 100.0);
        printf("  F1-Score: %.2f%%\n", f1 * 100.0);
        printf("  Temps: %.2f s\n\n", training_time);
        
        // Libérer le modèle
        free(predictions);
        free_logistic_regression(model);
    }
    
    // Sauvegarder les résultats
    save_experiment_results(results, NUM_LEARNING_RATES);
    
    // Afficher le résumé
    printf("\n================================================================================\n");
    printf("RÉSUMÉ DES RÉSULTATS\n");
    printf("================================================================================\n\n");
    printf("%-18s %-12s %-30s %-15s %-10s %-10s\n",
           "Learning rate (α)", "Iterations", "Convergence", "Accuracy Test", "F1-Score", "Temps (s)");
    printf("--------------------------------------------------------------------------------\n");
    for (int i = 0; i < NUM_LEARNING_RATES; i++) {
        printf("%-18.3f %-12d %-30s %-15.2f%% %-10.2f%% %-10.2f\n",
               results[i].learning_rate,
               results[i].iterations,
               results[i].convergence_status,
               results[i].test_accuracy * 100.0,
               results[i].test_f1 * 100.0,
               results[i].training_time);
    }
    printf("--------------------------------------------------------------------------------\n");
    
    // Libérer la mémoire
    free(results);
    free_dataset(train);
    free_dataset(test);
    free_scaler(scaler);
    
    printf("\n[OK] Expérimentation terminée!\n");
    
    return 0;
}

/* **************************************************
 * # --- MAIN (SEULEMENT SI COMPILÉ EN STANDALONE) --- #
 * ************************************************** */

#ifdef LEARNING_RATE_EXPERIMENT_STANDALONE
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    return run_learning_rate_experiment();
}
#endif

