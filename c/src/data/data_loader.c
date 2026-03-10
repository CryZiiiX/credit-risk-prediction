/*****************************************************************************************************

Nom : src/data/data_loader.c

Rôle : Chargement et gestion des datasets depuis fichiers CSV

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "data_loader.h"
#include <stdio.h>

// #include "../utils/memory_manager.h"

/* **************************************************
 * # --- CHARGEMENT DONNÉES --- #
 * ************************************************** */

/**
 * Fonction : load_train_data
 * Rôle     : Charge un dataset d'entraînement depuis un fichier CSV
 * Param    : filename (nom du fichier CSV à charger)
 * Retour   : Dataset* (structure Dataset contenant les données)
 */
Dataset* load_train_data(const char* filename) {
    return load_csv(filename, 1, -1);
}

/**
 * Fonction : load_test_data
 * Rôle     : Charge un dataset de test depuis un fichier CSV
 * Param    : filename (nom du fichier CSV à charger)
 * Retour   : Dataset* (structure Dataset contenant les données)
 */
Dataset* load_test_data(const char* filename) {
    return load_csv(filename, 1, -1);
}

/**
 * Fonction : save_dataset
 * Rôle     : Sauvegarde un dataset dans un fichier CSV
 * Param    : filename (nom du fichier de destination), dataset (dataset à sauvegarder)
 * Retour   : void
 */
void save_dataset(const char* filename, Dataset* dataset) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    // Write data
    for (int i = 0; i < dataset->rows; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            fprintf(file, "%.6f", dataset->data[i][j]);
            if (j < dataset->cols - 1) fprintf(file, ",");
        }
        if (dataset->labels) {
            fprintf(file, ",%d", dataset->labels[i]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}

/**
 * Fonction : save_dataset_with_header
 * Rôle     : Sauvegarde un dataset dans un fichier CSV avec header (compatible avec load_csv)
 * Param    : filename (nom du fichier de destination), dataset (dataset à sauvegarder)
 * Retour   : void
 */
void save_dataset_with_header(const char* filename, Dataset* dataset) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    // Écrire le header (noms de colonnes du dataset original)
    // Format: person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length
    // Le label (loan_status) est à l'index 8 dans le header
    fprintf(file, "person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length\n");
    
    // Write data
    // Le dataset->cols contient 11 features (sans le label)
    // Structure originale: 0-7 (8 features), 8 (label), 9-11 (3 features)
    // Dans dataset->data: indices 0-7 = features 0-7, indices 8-10 = features 9-11
    // On doit insérer le label à l'index 8 (après loan_int_rate, avant loan_percent_income)
    for (int i = 0; i < dataset->rows; i++) {
        // Écrire les 8 premières features (indices 0-7 dans dataset->data)
        for (int j = 0; j < 8 && j < dataset->cols; j++) {
            if (j > 0) fprintf(file, ",");
            fprintf(file, "%.6f", dataset->data[i][j]);
        }
        // Insérer le label à l'index 8
        fprintf(file, ",");
        if (dataset->labels) {
            fprintf(file, "%d", dataset->labels[i]);
        } else {
            fprintf(stderr, "ERROR: dataset->labels is NULL at row %d\n", i);
            fprintf(file, "0");
        }
        // Écrire les 3 dernières features (indices 8-10 dans dataset->data, qui correspondent aux features 9-11 originales)
        for (int j = 8; j < dataset->cols; j++) {
            fprintf(file, ",");
            fprintf(file, "%.6f", dataset->data[i][j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}
