/*****************************************************************************************************

Nom : src/preprocessing/scaler.c

Rôle : Implémentation StandardScaler pour normalisation des features numériques

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "scaler.h"
#include "../utils/memory_manager.h"
#include "../utils/utils.h"
#include <math.h>
#include <stdio.h>

/* **************************************************
 * # --- NORMALISATION STANDARDSCALER --- #
 * ************************************************** */

/**
 * Fonction : fit_scaler
 * Rôle     : Calcule la moyenne et l'écart-type de chaque feature pour la normalisation
 * Param    : dataset (dataset d'entraînement pour calculer les statistiques)
 * Retour   : Scaler* (scaler ajusté contenant moyenne et écart-type)
 */
Scaler* fit_scaler(Dataset* dataset) {
    Scaler* scaler = (Scaler*)safe_malloc(sizeof(Scaler));
    scaler->n_features = dataset->cols;
    scaler->mean = allocate_vector(dataset->cols);
    scaler->std = allocate_vector(dataset->cols);
    
    // Calculate mean
    for (int j = 0; j < dataset->cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < dataset->rows; i++) {
            sum += dataset->data[i][j];
        }
        scaler->mean[j] = sum / dataset->rows;
    }
    
    // Calculate std
    for (int j = 0; j < dataset->cols; j++) {
        double sum_sq = 0.0;
        for (int i = 0; i < dataset->rows; i++) {
            double diff = dataset->data[i][j] - scaler->mean[j];
            sum_sq += diff * diff;
        }
        scaler->std[j] = sqrt(sum_sq / dataset->rows);
        if (scaler->std[j] < 1e-8) scaler->std[j] = 1.0;
    }
    
    return scaler;
}

/**
 * Fonction : transform_dataset
 * Rôle     : Applique la normalisation StandardScaler (x' = (x - mean) / std) à un dataset
 * Param    : dataset (dataset à normaliser), scaler (scaler préalablement ajusté)
 * Retour   : void
 */
void transform_dataset(Dataset* dataset, Scaler* scaler) {
    for (int i = 0; i < dataset->rows; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            dataset->data[i][j] = (dataset->data[i][j] - scaler->mean[j]) / scaler->std[j];
        }
    }
}

/**
 * Fonction : save_scaler
 * Rôle     : Sauvegarde les paramètres d'un scaler dans un fichier texte
 * Param    : filename (nom du fichier de destination), scaler (scaler à sauvegarder)
 * Retour   : void
 */
void save_scaler(const char* filename, Scaler* scaler) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "%d\n", scaler->n_features);
    for (int i = 0; i < scaler->n_features; i++) {
        fprintf(file, "%.10f %.10f\n", scaler->mean[i], scaler->std[i]);
    }
    
    fclose(file);
}

/**
 * Fonction : load_scaler
 * Rôle     : Charge les paramètres d'un scaler depuis un fichier texte
 * Param    : filename (nom du fichier source)
 * Retour   : Scaler* (scaler chargé, NULL en cas d'erreur)
 */
Scaler* load_scaler(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    Scaler* scaler = (Scaler*)safe_malloc(sizeof(Scaler));
    if (fscanf(file, "%d", &scaler->n_features) != 1) {
        fclose(file);
        free(scaler);
        return NULL;
    }
    
    scaler->mean = allocate_vector(scaler->n_features);
    scaler->std = allocate_vector(scaler->n_features);
    
    for (int i = 0; i < scaler->n_features; i++) {
        if (fscanf(file, "%lf %lf", &scaler->mean[i], &scaler->std[i]) != 2) {
            fclose(file);
            free_scaler(scaler);
            return NULL;
        }
    }
    
    fclose(file);
    return scaler;
}

/**
 * Fonction : free_scaler
 * Rôle     : Libère complètement la mémoire allouée pour un scaler
 * Param    : scaler (scaler à libérer)
 * Retour   : void
 */
void free_scaler(Scaler* scaler) {
    if (scaler) {
        free_vector(scaler->mean);
        free_vector(scaler->std);
        safe_free(scaler);
    }
}

