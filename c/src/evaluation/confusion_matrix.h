/*****************************************************************************************************

Nom : src/evaluation/confusion_matrix.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de matrice de confusion

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

typedef struct {
    int tp;  // True Positives
    int tn;  // True Negatives
    int fp;  // False Positives
    int fn;  // False Negatives
} ConfusionMatrix;

ConfusionMatrix* compute_confusion_matrix(int* y_true, int* y_pred, int n_samples);
void print_confusion_matrix(ConfusionMatrix* cm);
void save_confusion_matrix(const char* filename, ConfusionMatrix* cm);
void free_confusion_matrix(ConfusionMatrix* cm);

#endif

