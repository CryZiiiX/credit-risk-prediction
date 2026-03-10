/*****************************************************************************************************

Nom : src/evaluation/confusion_matrix.c

Rôle : Calcul et affichage de la matrice de confusion

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "confusion_matrix.h"
#include "../utils/memory_manager.h"
#include <stdio.h>

/* **************************************************
 * # --- MATRICE DE CONFUSION --- #
 * ************************************************** */

/**
 * Fonction : compute_confusion_matrix
 * Rôle     : Calcule la matrice de confusion en comptant TP, TN, FP, FN
 * Param    : y_true (labels réels), y_pred (labels prédits), n_samples (nombre d'échantillons)
 * Retour   : ConfusionMatrix* (matrice de confusion calculée)
 */
ConfusionMatrix* compute_confusion_matrix(int* y_true, int* y_pred, int n_samples) {
    ConfusionMatrix* cm = (ConfusionMatrix*)safe_malloc(sizeof(ConfusionMatrix));
    cm->tp = cm->tn = cm->fp = cm->fn = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) cm->tp++;
        else if (y_true[i] == 0 && y_pred[i] == 0) cm->tn++;
        else if (y_true[i] == 0 && y_pred[i] == 1) cm->fp++;
        else if (y_true[i] == 1 && y_pred[i] == 0) cm->fn++;
    }
    
    return cm;
}

/**
 * Fonction : print_confusion_matrix
 * Rôle     : Affiche la matrice de confusion sous forme de tableau sur la sortie standard
 * Param    : cm (matrice de confusion à afficher)
 * Retour   : void
 */
void print_confusion_matrix(ConfusionMatrix* cm) {
    printf("\nConfusion Matrix:\n");
    printf("                Predicted\n");
    printf("              Neg    Pos\n");
    printf("Actual Neg  %5d  %5d\n", cm->tn, cm->fp);
    printf("       Pos  %5d  %5d\n", cm->fn, cm->tp);
}

/**
 * Fonction : save_confusion_matrix
 * Rôle     : Sauvegarde la matrice de confusion dans un fichier texte
 * Param    : filename (nom du fichier de destination), cm (matrice de confusion à sauvegarder)
 * Retour   : void
 */
void save_confusion_matrix(const char* filename, ConfusionMatrix* cm) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "Confusion Matrix:\n");
    fprintf(file, "TN: %d, FP: %d\n", cm->tn, cm->fp);
    fprintf(file, "FN: %d, TP: %d\n", cm->fn, cm->tp);
    
    fclose(file);
}

/**
 * Fonction : free_confusion_matrix
 * Rôle     : Libère la mémoire allouée pour une matrice de confusion
 * Param    : cm (matrice de confusion à libérer)
 * Retour   : void
 */
void free_confusion_matrix(ConfusionMatrix* cm) {
    safe_free(cm);
}

