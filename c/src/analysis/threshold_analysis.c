/*****************************************************************************************************

Nom : src/analysis/threshold_analysis.c

Rôle : Programme autonome pour analyser les seuils optimaux de classification pour la régression logistique C

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make threshold_analysis

    Pour executer : ./build/threshold_analysis

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "../data/data_splitter.h"
#include "../preprocessing/preprocessing.h"
#include "../preprocessing/scaler.h"
#include "../models/logistic_regression.h"
#include "../evaluation/metrics.h"
#include "../evaluation/confusion_matrix.h"
#include "../utils/csv_parser.h"
#include "threshold_analysis.h"

/* **************************************************
 * # --- CONSTANTES DE COÛT --- #
 * ************************************************** */

#define MONTANT_MOYEN_PRET 9590.0
#define TAUX_PERTE 0.4346
#define TAUX_INTERET_MOYEN 0.1101  // 11.01% (moyenne de loan_int_rate)
#define COUT_FN (MONTANT_MOYEN_PRET * TAUX_PERTE)  // 4167.81
#define COUT_FP (MONTANT_MOYEN_PRET * TAUX_INTERET_MOYEN)  // 1055.86 (marge brute par prêt)

/* **************************************************
 * # --- FONCTIONS UTILITAIRES --- #
 * ************************************************** */

/**
 * Fonction : predict_with_threshold
 * Rôle     : Prédit les classes binaires avec un seuil personnalisé
 * Param    : probabilities (probabilités), n_samples (nombre d'échantillons), threshold (seuil)
 * Retour   : int* (tableau de prédictions binaires)
 */
int* predict_with_threshold(double* probabilities, int n_samples, double threshold) {
    int* predictions = (int*)malloc(n_samples * sizeof(int));
    if (!predictions) {
        fprintf(stderr, "Erreur allocation mémoire pour prédictions\n");
        return NULL;
    }
    
    for (int i = 0; i < n_samples; i++) {
        predictions[i] = (probabilities[i] >= threshold) ? 1 : 0;
    }
    
    return predictions;
}

/**
 * Fonction : calculate_metrics_for_threshold
 * Rôle     : Calcule toutes les métriques pour un seuil donné
 * Param    : y_true (labels réels), probabilities (probabilités), n_samples (nombre d'échantillons), threshold (seuil)
 * Retour   : ThresholdResult (structure contenant toutes les métriques)
 */
ThresholdResult calculate_metrics_for_threshold(int* y_true, double* probabilities, int n_samples, double threshold) {
    ThresholdResult result;
    result.threshold = threshold;
    
    // Prédictions avec le seuil
    int* predictions = predict_with_threshold(probabilities, n_samples, threshold);
    if (!predictions) {
        // Valeurs par défaut en cas d'erreur
        result.accuracy = 0.0;
        result.precision = 0.0;
        result.recall = 0.0;
        result.f1 = 0.0;
        result.fp = 0;
        result.fn = 0;
        result.cost = 0.0;
        return result;
    }
    
    // Calculer les métriques
    result.accuracy = compute_accuracy(y_true, predictions, n_samples);
    result.precision = compute_precision(y_true, predictions, n_samples);
    result.recall = compute_recall(y_true, predictions, n_samples);
    result.f1 = compute_f1_score(y_true, predictions, n_samples);
    
    // Calculer FP et FN via la matrice de confusion
    ConfusionMatrix* cm = compute_confusion_matrix(y_true, predictions, n_samples);
    result.fp = cm->fp;
    result.fn = cm->fn;
    
    // Calculer le coût total
    result.cost = (result.fp * COUT_FP) + (result.fn * COUT_FN);
    
    // Libérer la mémoire
    free(predictions);
    free_confusion_matrix(cm);
    
    return result;
}

/**
 * Fonction : load_test_dataset
 * Rôle     : Charge le test set depuis le fichier sauvegardé ou le recrée
 * Param    : aucun
 * Retour   : Dataset* (test set prétraité et normalisé)
 */
Dataset* load_test_dataset(void) {
    // Essayer d'abord de charger le test set sauvegardé
    printf("Tentative de chargement du test set sauvegardé...\n");
    Dataset* test_set = load_csv_numeric("data/processed/test.csv", 1, 8);
    
    if (test_set && test_set->rows > 0) {
        printf("[OK] Test set chargé depuis data/processed/test.csv: %d échantillons\n", test_set->rows);
        
        // Le test set est déjà prétraité (encodage catégoriel fait), il faut juste appliquer la normalisation
        printf("\nNormalisation des données...\n");
        Scaler* scaler = load_scaler("data/processed/scaler_params.txt");
        if (!scaler) {
            fprintf(stderr, "Erreur: Impossible de charger le scaler\n");
            free_dataset(test_set);
            return NULL;
        }
        transform_dataset(test_set, scaler);
        printf("[OK] Normalisation terminée\n");
        free_scaler(scaler);
        
        return test_set;
    }
    
    // Si le fichier n'existe pas ou est vide, recréer le split
    printf("[INFO] Fichier test.csv non trouvé ou vide. Création d'un nouveau split...\n");
    
    printf("Chargement du dataset complet...\n");
    Dataset* dataset = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);
    if (!dataset) {
        fprintf(stderr, "Erreur: Impossible de charger le dataset\n");
        return NULL;
    }
    printf("[OK] Dataset chargé: %d échantillons, %d features\n", dataset->rows, dataset->cols);
    
    printf("\nPrétraitement des données...\n");
    preprocess_dataset(dataset);
    printf("[OK] Prétraitement terminé\n");
    
    printf("\nDivision train/test (80/20)...\n");
    // Utiliser srand(42) pour reproductibilité
    srand(42);
    shuffle_dataset(dataset);
    SplitData* split = split_dataset(dataset, 0.8);
    printf("[OK] Train: %d échantillons, Test: %d échantillons\n", split->train->rows, split->test->rows);
    
    printf("\nNormalisation des données...\n");
    // Charger le scaler sauvegardé
    Scaler* scaler = load_scaler("data/processed/scaler_params.txt");
    if (!scaler) {
        fprintf(stderr, "Erreur: Impossible de charger le scaler. Utilisation d'un nouveau scaler.\n");
        scaler = fit_scaler(split->train);
    }
    transform_dataset(split->test, scaler);
    printf("[OK] Normalisation terminée\n");
    
    // Libérer le train set et le dataset original (on garde seulement le test)
    free_dataset(split->train);
    free_dataset(dataset);
    free_scaler(scaler);
    
    // Retourner le test set
    Dataset* test_set_new = split->test;
    free(split);  // Libérer la structure SplitData mais garder le test set
    
    return test_set_new;
}

/**
 * Fonction : generate_threshold_table
 * Rôle     : Génère le tableau d'analyse des seuils optimaux
 * Param    : y_true (labels réels), probabilities (probabilités), n_samples (nombre d'échantillons)
 * Retour   : void
 */
void generate_threshold_table(int* y_true, double* probabilities, int n_samples) {
    printf("\n================================================================================\n");
    printf("ANALYSE DU SEUIL OPTIMAL - REGRESSION LOGISTIQUE C\n");
    printf("================================================================================\n\n");
    
    // Seuils à tester
    double thresholds[] = {0.3, 0.4, 0.5, 0.6, 0.7};
    int n_thresholds = 5;
    
    ThresholdResult* results = (ThresholdResult*)malloc(n_thresholds * sizeof(ThresholdResult));
    if (!results) {
        fprintf(stderr, "Erreur allocation mémoire pour résultats\n");
        return;
    }
    
    printf("Calcul des métriques pour chaque seuil...\n");
    for (int i = 0; i < n_thresholds; i++) {
        results[i] = calculate_metrics_for_threshold(y_true, probabilities, n_samples, thresholds[i]);
        printf("[OK] Seuil %.1f: Coût = %.2fM$\n", thresholds[i], results[i].cost / 1e6);
    }
    
    // Créer le répertoire de sortie
    int ret = system("mkdir -p results/metrics/logistic_regression");
    if (ret != 0) {
        fprintf(stderr, "Warning: Impossible de créer le répertoire results/metrics/logistic_regression\n");
    }
    
    // Ouvrir le fichier de sortie
    FILE* output_file = fopen("results/metrics/logistic_regression/lr_c_threshold_analysis.txt", "w");
    if (!output_file) {
        fprintf(stderr, "Erreur: Impossible de créer le fichier de sortie\n");
        free(results);
        return;
    }
    
    // Écrire l'en-tête
    fprintf(output_file, "================================================================================\n");
    fprintf(output_file, "ANALYSE DU SEUIL OPTIMAL - REGRESSION LOGISTIQUE C\n");
    fprintf(output_file, "================================================================================\n\n");
    
    fprintf(output_file, "Coûts utilisés:\n");
    fprintf(output_file, "  - Montant moyen du prêt: %.2f$\n", MONTANT_MOYEN_PRET);
    fprintf(output_file, "  - Taux de perte: %.4f (%.2f%%)\n", TAUX_PERTE, TAUX_PERTE * 100);
    fprintf(output_file, "  - Coût FP (Faux Positif): %.2f$\n", COUT_FP);
    fprintf(output_file, "  - Coût FN (Faux Négatif): %.2f$\n", COUT_FN);
    fprintf(output_file, "  - Ratio FN/FP: %.2f:1\n\n", COUT_FN / COUT_FP);
    
    // Écrire le tableau
    fprintf(output_file, "--------------------------------------------------------------------------------\n");
    fprintf(output_file, "%-8s %-9s %-9s %-9s %-9s %-8s %-8s %-13s\n",
            "Seuil", "Accuracy", "Precision", "Recall", "F1", "FP", "FN", "Coût estimé");
    fprintf(output_file, "--------------------------------------------------------------------------------\n");
    
    // Afficher aussi à l'écran
    printf("\n================================================================================\n");
    printf("TABLEAU D'ANALYSE DU SEUIL OPTIMAL\n");
    printf("================================================================================\n\n");
    printf("Coûts utilisés:\n");
    printf("  - Montant moyen du prêt: %.2f$\n", MONTANT_MOYEN_PRET);
    printf("  - Taux de perte: %.4f (%.2f%%)\n", TAUX_PERTE, TAUX_PERTE * 100);
    printf("  - Coût FP (Faux Positif): %.2f$\n", COUT_FP);
    printf("  - Coût FN (Faux Négatif): %.2f$\n", COUT_FN);
    printf("  - Ratio FN/FP: %.2f:1\n\n", COUT_FN / COUT_FP);
    
    printf("--------------------------------------------------------------------------------\n");
    printf("%-8s %-9s %-9s %-9s %-9s %-8s %-8s %-13s\n",
           "Seuil", "Accuracy", "Precision", "Recall", "F1", "FP", "FN", "Coût estimé");
    printf("--------------------------------------------------------------------------------\n");
    
    // Trouver le seuil optimal (coût minimal)
    int optimal_idx = 0;
    double min_cost = results[0].cost;
    
    for (int i = 0; i < n_thresholds; i++) {
        // Écrire dans le fichier
        fprintf(output_file, "%-8.1f %-9.2f%% %-9.2f%% %-9.2f%% %-9.2f%% %-8d %-8d %-13.2fM$\n",
                results[i].threshold,
                results[i].accuracy * 100,
                results[i].precision * 100,
                results[i].recall * 100,
                results[i].f1 * 100,
                results[i].fp,
                results[i].fn,
                results[i].cost / 1e6);
        
        // Afficher à l'écran
        printf("%-8.1f %-9.2f%% %-9.2f%% %-9.2f%% %-9.2f%% %-8d %-8d %-13.2fM$\n",
               results[i].threshold,
               results[i].accuracy * 100,
               results[i].precision * 100,
               results[i].recall * 100,
               results[i].f1 * 100,
               results[i].fp,
               results[i].fn,
               results[i].cost / 1e6);
        
        // Mettre à jour le seuil optimal
        if (results[i].cost < min_cost) {
            min_cost = results[i].cost;
            optimal_idx = i;
        }
    }
    
    fprintf(output_file, "--------------------------------------------------------------------------------\n");
    fprintf(output_file, "\nSeuil optimal: %.1f (Coût minimal = %.2fM$)\n",
            results[optimal_idx].threshold, results[optimal_idx].cost / 1e6);
    
    printf("--------------------------------------------------------------------------------\n");
    printf("\n[OK] Seuil optimal: %.1f (Coût minimal = %.2fM$)\n",
           results[optimal_idx].threshold, results[optimal_idx].cost / 1e6);
    
    fclose(output_file);
    free(results);
    
    printf("\n[OK] Tableau sauvegardé: results/metrics/logistic_regression/lr_c_threshold_analysis.txt\n");
}

/* **************************************************
 * # --- FONCTION MAIN (SEULEMENT EN STANDALONE) --- #
 * ************************************************** */

#ifdef THRESHOLD_ANALYSIS_STANDALONE
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    printf("Démarrage de l'analyse des seuils optimaux...\n");
    printf("================================================================================\n\n");
    
    // Charger le modèle
    printf("Chargement du modèle...\n");
    LogisticRegression* model = load_model("results/models/logistic_model.bin");
    if (!model) {
        fprintf(stderr, "Erreur: Impossible de charger le modèle depuis results/models/logistic_model.bin\n");
        fprintf(stderr, "Assurez-vous d'avoir exécuté le programme principal d'abord.\n");
        return 1;
    }
    printf("[OK] Modèle chargé (%d features)\n", model->n_features);
    
    // Charger le test set
    Dataset* test_set = load_test_dataset();
    if (!test_set) {
        fprintf(stderr, "Erreur: Impossible de charger le test set\n");
        free_logistic_regression(model);
        return 1;
    }
    
    // Vérifier que le nombre de features correspond
    if (test_set->cols != model->n_features) {
        fprintf(stderr, "Erreur: Nombre de features incompatible (dataset: %d, modèle: %d)\n",
                test_set->cols, model->n_features);
        free_dataset(test_set);
        free_logistic_regression(model);
        return 1;
    }
    
    // Calculer les probabilités
    printf("\nCalcul des probabilités sur le test set...\n");
    double* probabilities = predict_proba(model, test_set);
    if (!probabilities) {
        fprintf(stderr, "Erreur: Impossible de calculer les probabilités\n");
        free_dataset(test_set);
        free_logistic_regression(model);
        return 1;
    }
    printf("[OK] %d probabilités calculées\n", test_set->rows);
    
    // Générer le tableau d'analyse
    generate_threshold_table(test_set->labels, probabilities, test_set->rows);
    
    // Libérer la mémoire
    free(probabilities);
    free_dataset(test_set);
    free_logistic_regression(model);
    
    printf("\n[OK] Analyse terminée!\n");
    
    return 0;
}
#endif

/* **************************************************
 * # --- FONCTION WRAPPER POUR MAIN.C --- #
 * ************************************************** */

/**
 * Fonction : generate_threshold_analysis_table
 * Rôle     : Fonction wrapper pour générer l'analyse des seuils depuis main.c
 * Param    : model (modèle de régression logistique), test_set (dataset de test)
 * Retour   : void
 */
void generate_threshold_analysis_table(LogisticRegression* model, Dataset* test_set) {
    if (!model || !test_set) {
        fprintf(stderr, "Erreur: Modèle ou test set invalide\n");
        return;
    }
    
    // Calculer les probabilités
    double* probabilities = predict_proba(model, test_set);
    if (!probabilities) {
        fprintf(stderr, "Erreur: Impossible de calculer les probabilités\n");
        return;
    }
    
    // Générer le tableau
    generate_threshold_table(test_set->labels, probabilities, test_set->rows);
    
    // Libérer la mémoire
    free(probabilities);
}

