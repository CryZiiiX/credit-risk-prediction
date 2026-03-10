/*****************************************************************************************************

Nom : src/data/prepare_dataset.c

Rôle : Programme standalone pour générer le split train/test et les fichiers de préparation

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make prepare_dataset

    Pour executer : ./build/prepare_dataset

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "prepare_dataset.h"
#include "data_loader.h"
#include "data_splitter.h"
#include "../preprocessing/preprocessing.h"
#include "../preprocessing/scaler.h"
#include "../utils/csv_parser.h"

/* **************************************************
 * # --- FONCTION PRINCIPALE --- #
 * ************************************************** */

/**
 * Fonction : prepare_dataset
 * Rôle     : Génère le split train/test et sauvegarde les fichiers nécessaires
 * Param    : aucun
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int prepare_dataset(void) {
    printf("================================================================================\n");
    printf("PRÉPARATION DU DATASET - SPLIT TRAIN/TEST\n");
    printf("================================================================================\n\n");
    
    /* **************************************************
     * # --- CHARGEMENT DATASET --- #
     * ************************************************** */
    
    printf("Chargement du dataset...\n");
    Dataset* dataset = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);
    if (!dataset) {
        fprintf(stderr, "[ERREUR] Impossible de charger le dataset depuis data/raw/credit_risk_dataset.csv\n");
        return 1;
    }
    printf("[OK] Dataset chargé: %d échantillons, %d features\n", dataset->rows, dataset->cols);
    
    // Afficher statistiques des classes
    int class_0 = 0, class_1 = 0;
    for (int i = 0; i < dataset->rows; i++) {
        if (dataset->labels[i] == 0) class_0++;
        else class_1++;
    }
    printf("Balance des classes: Classe 0 (pas de défaut) = %d (%.1f%%), Classe 1 (défaut) = %d (%.1f%%)\n\n",
           class_0, 100.0 * class_0 / dataset->rows,
           class_1, 100.0 * class_1 / dataset->rows);
    
    /* **************************************************
     * # --- PRÉTRAITEMENT --- #
     * ************************************************** */
    
    printf("Prétraitement des données...\n");
    preprocess_dataset(dataset);
    printf("[OK] Prétraitement terminé\n\n");
    
    /* **************************************************
     * # --- DIVISION TRAIN/TEST --- #
     * ************************************************** */
    
    printf("Division train/test (80%% train, 20%% test)...\n");
    // Utiliser srand(42) pour reproductibilité
    srand(42);
    shuffle_dataset(dataset);
    SplitData* split = split_dataset(dataset, 0.8);
    printf("[OK] Train set: %d échantillons\n", split->train->rows);
    printf("[OK] Test set: %d échantillons\n\n", split->test->rows);
    
    /* **************************************************
     * # --- SAUVEGARDE DES DATASETS --- #
     * ************************************************** */
    
    printf("Sauvegarde des datasets...\n");
    int ret = system("mkdir -p data/processed");
    if (ret != 0) {
        fprintf(stderr, "[WARNING] Impossible de créer le répertoire data/processed\n");
    }
    
    // Sauvegarder les datasets (après prétraitement, avant normalisation)
    save_dataset_with_header("data/processed/train.csv", split->train);
    save_dataset_with_header("data/processed/test.csv", split->test);
    printf("[OK] Train set sauvegardé: data/processed/train.csv\n");
    printf("[OK] Test set sauvegardé: data/processed/test.csv\n\n");
    
    /* **************************************************
     * # --- CALCUL ET SAUVEGARDE DU SCALER --- #
     * ************************************************** */
    
    printf("Calcul du scaler sur le train set...\n");
    Scaler* scaler = fit_scaler(split->train);
    save_scaler("data/processed/scaler_params.txt", scaler);
    printf("[OK] Scaler sauvegardé: data/processed/scaler_params.txt\n\n");
    
    // Libérer la mémoire
    free_scaler(scaler);
    free_dataset(split->train);
    free_dataset(split->test);
    free(split);
    free_dataset(dataset);
    
    /* **************************************************
     * # --- SAUVEGARDE DES MÉTADONNÉES --- #
     * ************************************************** */
    
    printf("Sauvegarde des métadonnées...\n");
    FILE* metadata = fopen("data/processed/split_info.txt", "w");
    if (metadata) {
        fprintf(metadata, "================================================================================\n");
        fprintf(metadata, "INFORMATIONS SUR LE SPLIT\n");
        fprintf(metadata, "================================================================================\n\n");
        fprintf(metadata, "Fichier d'origine: data/raw/credit_risk_dataset.csv\n");
        fprintf(metadata, "Random state: 42\n");
        fprintf(metadata, "Test size: 0.2 (20%%)\n");
        fprintf(metadata, "Train size: 0.8 (80%%)\n\n");
        fprintf(metadata, "Fichiers générés:\n");
        fprintf(metadata, "  - data/processed/train.csv (prétraité, non normalisé)\n");
        fprintf(metadata, "  - data/processed/test.csv (prétraité, non normalisé)\n");
        fprintf(metadata, "  - data/processed/scaler_params.txt\n\n");
        fprintf(metadata, "Note: Les datasets sont prétraités (encodage catégoriel) mais pas normalisés.\n");
        fprintf(metadata, "      La normalisation sera appliquée lors de l'entraînement.\n");
        fclose(metadata);
        printf("[OK] Métadonnées sauvegardées: data/processed/split_info.txt\n\n");
    } else {
        fprintf(stderr, "[WARNING] Impossible de sauvegarder les métadonnées\n");
    }
    
    printf("================================================================================\n");
    printf("[OK] PRÉPARATION TERMINÉE AVEC SUCCÈS\n");
    printf("================================================================================\n\n");
    printf("Vous pouvez maintenant lancer l'entraînement avec:\n");
    printf("  ./build/credit_risk_predictor\n\n");
    
    return 0;
}

/* **************************************************
 * # --- MAIN (SEULEMENT SI COMPILÉ EN STANDALONE) --- #
 * ************************************************** */

#ifdef PREPARE_DATASET_STANDALONE
int main(void) {
    return prepare_dataset();
}
#endif

