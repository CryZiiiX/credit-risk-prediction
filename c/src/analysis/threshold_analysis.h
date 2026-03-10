/*****************************************************************************************************

Nom : src/analysis/threshold_analysis.h

Rôle : Déclarations pour le module d'analyse des seuils optimaux

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make threshold_analysis

    Pour executer : ./build/threshold_analysis

******************************************************************************************************/

#ifndef THRESHOLD_ANALYSIS_H
#define THRESHOLD_ANALYSIS_H

#include "../utils/csv_parser.h"
#include "../models/logistic_regression.h"

typedef struct {
    double threshold;
    double accuracy;
    double precision;
    double recall;
    double f1;
    int fp;
    int fn;
    double cost;
} ThresholdResult;

int* predict_with_threshold(double* probabilities, int n_samples, double threshold);
ThresholdResult calculate_metrics_for_threshold(int* y_true, double* probabilities, int n_samples, double threshold);
Dataset* load_test_dataset(void);
void generate_threshold_table(int* y_true, double* probabilities, int n_samples);

// Fonction wrapper pour être appelée depuis main.c
void generate_threshold_analysis_table(LogisticRegression* model, Dataset* test_set);

#endif

