/*****************************************************************************************************

Nom : src/models/logistic_regression.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de régression logistique

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "../utils/csv_parser.h"

typedef struct {
    double* weights;
    double bias;
    int n_features;
    double learning_rate;
    int max_iterations;
} LogisticRegression;

LogisticRegression* create_logistic_regression(int n_features, double learning_rate, int max_iterations);
void train_logistic_regression(LogisticRegression* model, Dataset* dataset);
double train_logistic_regression_with_cost(LogisticRegression* model, Dataset* dataset);
int* predict(LogisticRegression* model, Dataset* dataset);
double* predict_proba(LogisticRegression* model, Dataset* dataset);
void save_model(const char* filename, LogisticRegression* model);
LogisticRegression* load_model(const char* filename);
void free_logistic_regression(LogisticRegression* model);
void generate_convergence_table(const char* csv_filename, const char* output_filename);

#endif
