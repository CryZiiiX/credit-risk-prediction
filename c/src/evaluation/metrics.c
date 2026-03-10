/*****************************************************************************************************

Nom : src/evaluation/metrics.c

Rôle : Calcul des métriques (accuracy, precision, recall, F1-score, AUC-ROC)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* **************************************************
 * # --- MÉTRIQUES DE CLASSIFICATION --- #
 * ************************************************** */

/**
 * Fonction : compute_accuracy
 * Rôle     : Calcule le taux de précision (proportion de prédictions correctes)
 * Param    : y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : double (accuracy entre 0 et 1)
 */
double compute_accuracy(int* y_true, int* y_pred, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return (double)correct / n_samples;
}

/**
 * Fonction : compute_precision
 * Rôle     : Calcule la précision (proportion de vrais positifs parmi les prédictions positives)
 * Param    : y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : double (precision entre 0 et 1)
 */
double compute_precision(int* y_true, int* y_pred, int n_samples) {
    int tp = 0, fp = 0;
    for (int i = 0; i < n_samples; i++) {
        if (y_pred[i] == 1) {
            if (y_true[i] == 1) tp++;
            else fp++;
        }
    }
    return (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
}

/**
 * Fonction : compute_recall
 * Rôle     : Calcule le rappel (proportion de vrais positifs détectés parmi les positifs réels)
 * Param    : y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : double (recall entre 0 et 1)
 */
double compute_recall(int* y_true, int* y_pred, int n_samples) {
    int tp = 0, fn = 0;
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1) {
            if (y_pred[i] == 1) tp++;
            else fn++;
        }
    }
    return (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
}

/**
 * Fonction : compute_f1_score
 * Rôle     : Calcule le score F1 (moyenne harmonique de la précision et du rappel)
 * Param    : y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : double (F1-score entre 0 et 1)
 */
double compute_f1_score(int* y_true, int* y_pred, int n_samples) {
    double precision = compute_precision(y_true, y_pred, n_samples);
    double recall = compute_recall(y_true, y_pred, n_samples);
    
    if (precision + recall == 0) return 0.0;
    return 2 * (precision * recall) / (precision + recall);
}

/**
 * Fonction : print_metrics
 * Rôle     : Affiche toutes les métriques de classification sur la sortie standard
 * Param    : y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : void
 */
void print_metrics(int* y_true, int* y_pred, int n_samples) {
    printf("Accuracy:  %.4f\n", compute_accuracy(y_true, y_pred, n_samples));
    printf("Precision: %.4f\n", compute_precision(y_true, y_pred, n_samples));
    printf("Recall:    %.4f\n", compute_recall(y_true, y_pred, n_samples));
    printf("F1-Score:  %.4f\n", compute_f1_score(y_true, y_pred, n_samples));
}

/**
 * Fonction : save_metrics
 * Rôle     : Sauvegarde toutes les métriques de classification dans un fichier texte
 * Param    : filename (nom du fichier de destination), y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : void
 */
void save_metrics(const char* filename, int* y_true, int* y_pred, int n_samples) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "Accuracy: %.6f\n", compute_accuracy(y_true, y_pred, n_samples));
    fprintf(file, "Precision: %.6f\n", compute_precision(y_true, y_pred, n_samples));
    fprintf(file, "Recall: %.6f\n", compute_recall(y_true, y_pred, n_samples));
    fprintf(file, "F1-Score: %.6f\n", compute_f1_score(y_true, y_pred, n_samples));
    
    fclose(file);
}

/**
 * Fonction : save_metrics_with_auc
 * Rôle     : Sauvegarde toutes les métriques de classification incluant l'AUC-ROC dans un fichier texte
 * Param    : filename (nom du fichier de destination), y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons), auc_roc (valeur AUC-ROC)
 * Retour   : void
 */
void save_metrics_with_auc(const char* filename, int* y_true, int* y_pred, int n_samples, double auc_roc) {
    save_metrics_with_auc_and_title(filename, NULL, y_true, y_pred, n_samples, auc_roc);
}

/**
 * Fonction : save_metrics_with_auc_and_title
 * Rôle     : Sauvegarde toutes les métriques de classification incluant l'AUC-ROC dans un fichier texte avec un titre
 * Param    : filename (nom du fichier de destination), title (titre optionnel, NULL si absent), y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons), auc_roc (valeur AUC-ROC)
 * Retour   : void
 */
void save_metrics_with_auc_and_title(const char* filename, const char* title, int* y_true, int* y_pred, int n_samples, double auc_roc) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    // Ajouter un en-tête si un titre est fourni
    if (title != NULL) {
        fprintf(file, "============================================================\n");
        fprintf(file, "%s\n", title);
        fprintf(file, "============================================================\n\n");
    }
    
    fprintf(file, "Accuracy: %.6f\n", compute_accuracy(y_true, y_pred, n_samples));
    fprintf(file, "Precision: %.6f\n", compute_precision(y_true, y_pred, n_samples));
    fprintf(file, "Recall: %.6f\n", compute_recall(y_true, y_pred, n_samples));
    fprintf(file, "F1-Score: %.6f\n", compute_f1_score(y_true, y_pred, n_samples));
    fprintf(file, "AUC-ROC: %.6f\n", auc_roc);
    
    fclose(file);
}

/* **************************************************
 * # --- CALCUL AUC-ROC --- #
 * ************************************************** */

typedef struct {
    double probability;
    int true_label;
} PredictionPair;

/**
 * Fonction : compare_predictions_desc
 * Rôle     : Fonction de comparaison pour trier les prédictions par probabilité décroissante
 * Param    : a (pointeur vers première paire), b (pointeur vers deuxième paire)
 * Retour   : int (négatif si a > b, positif si a < b, 0 si égal)
 */
static int compare_predictions_desc(const void* a, const void* b) {
    PredictionPair* pair_a = (PredictionPair*)a;
    PredictionPair* pair_b = (PredictionPair*)b;
    
    if (pair_a->probability > pair_b->probability) return -1;
    if (pair_a->probability < pair_b->probability) return 1;
    return 0;
}

/**
 * Fonction : compute_auc_roc
 * Rôle     : Calcule l'aire sous la courbe ROC en triant les prédictions et calculant TPR/FPR pour chaque seuil
 * Param    : probabilities (probabilités de classe positive), y_true (labels réels), n_samples (nombre d'échantillons)
 * Retour   : double (AUC-ROC entre 0 et 1, 0.5 = performance aléatoire)
 */
double compute_auc_roc(double* probabilities, int* y_true, int n_samples) {
    /* Etape 1: Creer et trier les paires (probabilite, label) */
    PredictionPair* pairs = (PredictionPair*)malloc(n_samples * sizeof(PredictionPair));
    if (!pairs) {
        fprintf(stderr, "Erreur allocation memoire pour AUC-ROC\n");
        return 0.5;
    }
    
    for (int i = 0; i < n_samples; i++) {
        pairs[i].probability = probabilities[i];
        pairs[i].true_label = y_true[i];
    }
    
    qsort(pairs, n_samples, sizeof(PredictionPair), compare_predictions_desc);
    
    /* Etape 2: Compter le nombre de positifs et negatifs */
    int n_positive = 0;
    int n_negative = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1) n_positive++;
        else n_negative++;
    }
    
    /* Cas limite: pas de positifs ou pas de negatifs */
    if (n_positive == 0 || n_negative == 0) {
        free(pairs);
        return 0.5;
    }
    
    /* Etape 3: Allouer tableaux pour TPR et FPR */
    double* tpr_values = (double*)malloc(n_samples * sizeof(double));
    double* fpr_values = (double*)malloc(n_samples * sizeof(double));
    
    if (!tpr_values || !fpr_values) {
        fprintf(stderr, "Erreur allocation memoire pour TPR/FPR\n");
        free(pairs);
        if (tpr_values) free(tpr_values);
        if (fpr_values) free(fpr_values);
        return 0.5;
    }
    
    /* Etape 4: Calculer TPR et FPR pour chaque seuil */
    int tp = 0;
    int fp = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (pairs[i].true_label == 1) {
            tp++;
        } else {
            fp++;
        }
        
        tpr_values[i] = (double)tp / n_positive;
        fpr_values[i] = (double)fp / n_negative;
    }
    
    /* Etape 5: Calculer l'aire sous la courbe (methode des trapezes) */
    double auc = 0.0;
    
    for (int i = 1; i < n_samples; i++) {
        double width = fpr_values[i] - fpr_values[i-1];
        double avg_height = (tpr_values[i] + tpr_values[i-1]) / 2.0;
        auc += width * avg_height;
    }
    
    /* Nettoyage */
    free(pairs);
    free(tpr_values);
    free(fpr_values);
    
    return auc;
}

/**
 * Fonction : compute_auc_roc_and_save
 * Rôle     : Calcule l'aire sous la courbe ROC et sauvegarde les valeurs FPR/TPR dans un fichier CSV
 * Param    : probabilities (probabilités de classe positive), y_true (labels réels), n_samples (nombre d'échantillons), filename (nom du fichier CSV de sortie)
 * Retour   : double (AUC-ROC entre 0 et 1, 0.5 = performance aléatoire)
 */
double compute_auc_roc_and_save(double* probabilities, int* y_true, int n_samples, const char* filename) {
    /* Etape 1: Creer et trier les paires (probabilite, label) */
    PredictionPair* pairs = (PredictionPair*)malloc(n_samples * sizeof(PredictionPair));
    if (!pairs) {
        fprintf(stderr, "Erreur allocation memoire pour AUC-ROC\n");
        return 0.5;
    }
    
    for (int i = 0; i < n_samples; i++) {
        pairs[i].probability = probabilities[i];
        pairs[i].true_label = y_true[i];
    }
    
    qsort(pairs, n_samples, sizeof(PredictionPair), compare_predictions_desc);
    
    /* Etape 2: Compter le nombre de positifs et negatifs */
    int n_positive = 0;
    int n_negative = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1) n_positive++;
        else n_negative++;
    }
    
    /* Cas limite: pas de positifs ou pas de negatifs */
    if (n_positive == 0 || n_negative == 0) {
        free(pairs);
        return 0.5;
    }
    
    /* Etape 3: Allouer tableaux pour TPR et FPR */
    double* tpr_values = (double*)malloc(n_samples * sizeof(double));
    double* fpr_values = (double*)malloc(n_samples * sizeof(double));
    
    if (!tpr_values || !fpr_values) {
        fprintf(stderr, "Erreur allocation memoire pour TPR/FPR\n");
        free(pairs);
        if (tpr_values) free(tpr_values);
        if (fpr_values) free(fpr_values);
        return 0.5;
    }
    
    /* Etape 4: Calculer TPR et FPR pour chaque seuil */
    int tp = 0;
    int fp = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (pairs[i].true_label == 1) {
            tp++;
        } else {
            fp++;
        }
        
        tpr_values[i] = (double)tp / n_positive;
        fpr_values[i] = (double)fp / n_negative;
    }
    
    /* Etape 5: Sauvegarder les valeurs FPR/TPR dans le fichier CSV */
    if (filename != NULL) {
        FILE* roc_file = fopen(filename, "w");
        if (roc_file) {
            fprintf(roc_file, "fpr,tpr\n");
            
            /* Sauvegarder tous les points (commence à (0,0) implicitement) */
            /* Ajouter le point initial (0,0) */
            fprintf(roc_file, "0.000000,0.000000\n");
            
            /* Sauvegarder les points calculés, en évitant les doublons consécutifs */
            double last_fpr = -1.0;
            double last_tpr = -1.0;
            
            for (int i = 0; i < n_samples; i++) {
                /* Éviter les doublons consécutifs */
                if (fpr_values[i] != last_fpr || tpr_values[i] != last_tpr) {
                    fprintf(roc_file, "%.6f,%.6f\n", fpr_values[i], tpr_values[i]);
                    last_fpr = fpr_values[i];
                    last_tpr = tpr_values[i];
                }
            }
            
            /* S'assurer que le dernier point est (1,1) */
            if (last_fpr < 1.0 || last_tpr < 1.0) {
                fprintf(roc_file, "1.000000,1.000000\n");
            }
            
            fclose(roc_file);
        } else {
            fprintf(stderr, "Warning: Cannot create ROC data file: %s\n", filename);
        }
    }
    
    /* Etape 6: Calculer l'aire sous la courbe (methode des trapezes) */
    double auc = 0.0;
    
    for (int i = 1; i < n_samples; i++) {
        double width = fpr_values[i] - fpr_values[i-1];
        double avg_height = (tpr_values[i] + tpr_values[i-1]) / 2.0;
        auc += width * avg_height;
    }
    
    /* Nettoyage */
    free(pairs);
    free(tpr_values);
    free(fpr_values);
    
    return auc;
}

