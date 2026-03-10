/*****************************************************************************************************

Nom : src/analysis/learning_rate_experiment.h

Rôle : Déclarations pour le module d'expérimentation du learning rate

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make learning_rate_experiment

    Pour executer : ./build/learning_rate_experiment

******************************************************************************************************/

#ifndef LEARNING_RATE_EXPERIMENT_H
#define LEARNING_RATE_EXPERIMENT_H

#include "../models/logistic_regression.h"
#include "../utils/csv_parser.h"

/**
 * Structure pour stocker les résultats d'une expérience avec un learning rate
 */
typedef struct {
    double learning_rate;
    int iterations;
    char convergence_status[64];  // "Oui", "Non", "Partielle", "Oscillations", "Divergence"
    double final_cost;
    double test_accuracy;
    double test_f1;
    double training_time;
} LearningRateResult;

/**
 * Fonction : run_learning_rate_experiment
 * Rôle     : Lance l'expérimentation complète avec différents learning rates
 * Param    : aucun
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int run_learning_rate_experiment(void);

#endif

