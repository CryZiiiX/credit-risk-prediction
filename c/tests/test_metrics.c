/*****************************************************************************************************

Nom : tests/test_metrics.c

Rôle : Tests unitaires pour le module metrics

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./test_metrics

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/evaluation/metrics.h"
#include "../src/evaluation/confusion_matrix.h"

void test_perfect_predictions() {
    printf("Test 1: Prédictions parfaites... ");
    
    int y_true[] = {0, 0, 1, 1, 0, 1};
    int y_pred[] = {0, 0, 1, 1, 0, 1};
    int n = 6;
    
    double acc = compute_accuracy(y_true, y_pred, n);
    double prec = compute_precision(y_true, y_pred, n);
    double rec = compute_recall(y_true, y_pred, n);
    double f1 = compute_f1_score(y_true, y_pred, n);
    
    assert(fabs(acc - 1.0) < 1e-6);
    assert(fabs(prec - 1.0) < 1e-6);
    assert(fabs(rec - 1.0) < 1e-6);
    assert(fabs(f1 - 1.0) < 1e-6);
    
    printf("[OK] PASSÉ\n");
}

void test_all_wrong() {
    printf("Test 2: Toutes les prédictions fausses... ");
    
    int y_true[] = {0, 0, 1, 1};
    int y_pred[] = {1, 1, 0, 0};
    int n = 4;
    
    double acc = compute_accuracy(y_true, y_pred, n);
    
    assert(fabs(acc - 0.0) < 1e-6);
    
    printf("[OK] PASSÉ\n");
}

void test_mixed_predictions() {
    printf("Test 3: Prédictions mixtes... ");
    
    int y_true[] = {0, 0, 0, 1, 1, 1};
    int y_pred[] = {0, 0, 1, 1, 1, 0};
    int n = 6;
    
    // TP=2, TN=2, FP=1, FN=1
    // Accuracy = (2+2)/6 = 0.666...
    // Precision = 2/(2+1) = 0.666...
    // Recall = 2/(2+1) = 0.666...
    // F1 = 2 * 0.666 * 0.666 / (0.666 + 0.666) = 0.666...
    
    double acc = compute_accuracy(y_true, y_pred, n);
    double prec = compute_precision(y_true, y_pred, n);
    double rec = compute_recall(y_true, y_pred, n);
    double f1 = compute_f1_score(y_true, y_pred, n);
    
    assert(fabs(acc - 0.6666666) < 0.01);
    assert(fabs(prec - 0.6666666) < 0.01);
    assert(fabs(rec - 0.6666666) < 0.01);
    assert(fabs(f1 - 0.6666666) < 0.01);
    
    printf("[OK] PASSÉ\n");
}

void test_no_positive_predictions() {
    printf("Test 4: Aucune prédiction positive... ");
    
    int y_true[] = {0, 0, 1, 1};
    int y_pred[] = {0, 0, 0, 0};
    int n = 4;
    
    double prec = compute_precision(y_true, y_pred, n);
    double rec = compute_recall(y_true, y_pred, n);
    
    // Precision devrait être 0 (pas de TP, pas de FP)
    // Recall devrait être 0 (TP=0)
    assert(fabs(prec - 0.0) < 1e-6);
    assert(fabs(rec - 0.0) < 1e-6);
    
    printf("[OK] PASSÉ\n");
}

void test_confusion_matrix() {
    printf("Test 5: Matrice de confusion... ");
    
    int y_true[] = {0, 0, 0, 1, 1, 1, 1};
    int y_pred[] = {0, 0, 1, 1, 1, 0, 1};
    int n = 7;
    
    // TN=2, FP=1, FN=1, TP=3
    ConfusionMatrix* cm = compute_confusion_matrix(y_true, y_pred, n);
    
    assert(cm->tn == 2);
    assert(cm->fp == 1);
    assert(cm->fn == 1);
    assert(cm->tp == 3);
    
    free_confusion_matrix(cm);
    
    printf("[OK] PASSÉ\n");
}

void test_save_metrics() {
    printf("Test 6: Sauvegarde des métriques... ");
    
    int y_true[] = {0, 1, 0, 1};
    int y_pred[] = {0, 1, 0, 1};
    int n = 4;
    
    save_metrics("test_metrics.txt", y_true, y_pred, n);
    
    // Vérifier que le fichier existe et contient des données
    FILE* f = fopen("test_metrics.txt", "r");
    assert(f != NULL);
    
    char line[256];
    int lines = 0;
    while (fgets(line, sizeof(line), f)) {
        lines++;
    }
    
    assert(lines >= 4);  // Au moins 4 lignes (accuracy, precision, recall, f1)
    
    fclose(f);
    remove("test_metrics.txt");
    
    printf("[OK] PASSÉ\n");
}

void test_save_confusion_matrix() {
    printf("Test 7: Sauvegarde de la matrice de confusion... ");
    
    ConfusionMatrix cm;
    cm.tp = 10;
    cm.tn = 20;
    cm.fp = 5;
    cm.fn = 3;
    
    save_confusion_matrix("test_cm.txt", &cm);
    
    FILE* f = fopen("test_cm.txt", "r");
    assert(f != NULL);
    fclose(f);
    
    remove("test_cm.txt");
    
    printf("[OK] PASSÉ\n");
}

void test_auc_roc_perfect() {
    printf("Test 8: AUC-ROC avec classification parfaite... ");
    
    /* Dataset parfait : toutes les probabilites des positifs > probabilites des negatifs */
    double probabilities[] = {0.1, 0.2, 0.3, 0.7, 0.8, 0.9};
    int y_true[] = {0, 0, 0, 1, 1, 1};
    int n = 6;
    
    double auc = compute_auc_roc(probabilities, y_true, n);
    
    /* AUC devrait etre 1.0 (classification parfaite) */
    assert(fabs(auc - 1.0) < 0.01);
    
    printf("[OK] PASSÉ (AUC=%.4f)\n", auc);
}

void test_auc_roc_random() {
    printf("Test 9: AUC-ROC avec classification intermediaire... ");
    
    /* Classification intermediaire : quelques bonnes, quelques mauvaises predictions */
    double probabilities[] = {0.2, 0.6, 0.3, 0.4, 0.7, 0.5};
    int y_true[] = {0, 1, 0, 0, 1, 1};
    int n = 6;
    
    double auc = compute_auc_roc(probabilities, y_true, n);
    
    /* AUC devrait etre entre 0.0 et 1.0 (test general) */
    assert(auc >= 0.0 && auc <= 1.0);
    
    printf("[OK] PASSÉ (AUC=%.4f)\n", auc);
}

void test_auc_roc_intermediate() {
    printf("Test 10: AUC-ROC avec classification intermediaire... ");
    
    /* Cas intermediaire avec quelques erreurs */
    double probabilities[] = {0.1, 0.4, 0.35, 0.8, 0.6, 0.9};
    int y_true[] = {0, 0, 1, 1, 0, 1};
    int n = 6;
    
    double auc = compute_auc_roc(probabilities, y_true, n);
    
    /* AUC devrait etre entre 0.5 et 1.0 */
    assert(auc > 0.5 && auc < 1.0);
    
    printf("[OK] PASSÉ (AUC=%.4f)\n", auc);
}

void test_auc_roc_all_positive() {
    printf("Test 11: AUC-ROC avec toutes les classes positives... ");
    
    /* Cas limite : tous les labels sont 1 */
    double probabilities[] = {0.1, 0.5, 0.9};
    int y_true[] = {1, 1, 1};
    int n = 3;
    
    double auc = compute_auc_roc(probabilities, y_true, n);
    
    /* AUC devrait etre 0.5 (indefini, retourne 0.5) */
    assert(fabs(auc - 0.5) < 1e-6);
    
    printf("[OK] PASSÉ (AUC=%.4f, cas limite)\n", auc);
}

void test_auc_roc_all_negative() {
    printf("Test 12: AUC-ROC avec toutes les classes negatives... ");
    
    /* Cas limite : tous les labels sont 0 */
    double probabilities[] = {0.1, 0.5, 0.9};
    int y_true[] = {0, 0, 0};
    int n = 3;
    
    double auc = compute_auc_roc(probabilities, y_true, n);
    
    /* AUC devrait etre 0.5 (indefini, retourne 0.5) */
    assert(fabs(auc - 0.5) < 1e-6);
    
    printf("[OK] PASSÉ (AUC=%.4f, cas limite)\n", auc);
}

int main() {
    printf("\n=== TESTS DES MÉTRIQUES ===\n\n");
    
    test_perfect_predictions();
    test_all_wrong();
    test_mixed_predictions();
    test_no_positive_predictions();
    test_confusion_matrix();
    test_save_metrics();
    test_save_confusion_matrix();
    test_auc_roc_perfect();
    test_auc_roc_random();
    test_auc_roc_intermediate();
    test_auc_roc_all_positive();
    test_auc_roc_all_negative();
    
    printf("\n[OK] Tous les tests sont passés avec succès!\n\n");
    return 0;
}

