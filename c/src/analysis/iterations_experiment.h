/*****************************************************************************************************

Nom : src/analysis/iterations_experiment.h

Rôle : Déclarations pour le module d'expérimentation du nombre d'itérations

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make iterations_experiment

    Pour executer : ./build/iterations_experiment

******************************************************************************************************/

#ifndef ITERATIONS_EXPERIMENT_H
#define ITERATIONS_EXPERIMENT_H

#include "../models/logistic_regression.h"
#include "../utils/csv_parser.h"

/**
 * Structure pour stocker les résultats d'une expérience avec un nombre d'itérations
 */
typedef struct {
    int iterations;
    char convergence_status[64];  // "Oui", "Non", "Partielle", "Oscillations", "Divergence"
    double final_cost;
    double test_accuracy;
    double test_f1;
    double training_time;
} IterationsResult;

/**
 * Fonction : run_iterations_experiment
 * Rôle     : Lance l'expérimentation complète avec différents nombres d'itérations
 * Param    : aucun
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int run_iterations_experiment(void);

#endif

