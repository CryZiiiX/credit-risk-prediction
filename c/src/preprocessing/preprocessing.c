/*****************************************************************************************************

Nom : src/preprocessing/preprocessing.c

Rôle : Pipeline de prétraitement (gestion valeurs manquantes, normalisation)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "preprocessing.h"
#include <math.h>

/* **************************************************
 * # --- PRÉTRAITEMENT --- #
 * ************************************************** */

/**
 * Fonction : handle_missing_values
 * Rôle     : Remplace les valeurs manquantes (NaN) par la moyenne de chaque colonne
 * Param    : dataset (dataset à traiter)
 * Retour   : void
 */
void handle_missing_values(Dataset* dataset) {
    // Replace NaN with column mean
    for (int j = 0; j < dataset->cols; j++) {
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < dataset->rows; i++) {
            if (!isnan(dataset->data[i][j])) {
                sum += dataset->data[i][j];
                count++;
            }
        }
        
        double mean = (count > 0) ? sum / count : 0.0;
        
        for (int i = 0; i < dataset->rows; i++) {
            if (isnan(dataset->data[i][j])) {
                dataset->data[i][j] = mean;
            }
        }
    }
}

/**
 * Fonction : normalize_features
 * Rôle     : Normalise les features d'un dataset à l'aide d'un scaler préalablement ajusté
 * Param    : dataset (dataset à normaliser), scaler (scaler contenant moyenne et écart-type)
 * Retour   : void
 */
void normalize_features(Dataset* dataset, Scaler* scaler) {
    transform_dataset(dataset, scaler);
}

/**
 * Fonction : preprocess_dataset
 * Rôle     : Applique le pipeline de prétraitement complet (gestion valeurs manquantes)
 * Param    : dataset (dataset à prétraiter)
 * Retour   : Dataset* (dataset prétraité)
 */
Dataset* preprocess_dataset(Dataset* dataset) {
    handle_missing_values(dataset);
    return dataset;
}

