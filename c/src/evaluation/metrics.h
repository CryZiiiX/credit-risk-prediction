/*****************************************************************************************************

Nom : src/evaluation/metrics.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de calcul de métriques

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef METRICS_H
#define METRICS_H

double compute_accuracy(int* y_true, int* y_pred, int n_samples);
double compute_precision(int* y_true, int* y_pred, int n_samples);
double compute_recall(int* y_true, int* y_pred, int n_samples);
double compute_f1_score(int* y_true, int* y_pred, int n_samples);
double compute_auc_roc(double* probabilities, int* y_true, int n_samples);
double compute_auc_roc_and_save(double* probabilities, int* y_true, int n_samples, const char* filename);
void print_metrics(int* y_true, int* y_pred, int n_samples);
void save_metrics(const char* filename, int* y_true, int* y_pred, int n_samples);
void save_metrics_with_auc(const char* filename, int* y_true, int* y_pred, int n_samples, double auc_roc);
void save_metrics_with_auc_and_title(const char* filename, const char* title, int* y_true, int* y_pred, int n_samples, double auc_roc);

#endif
