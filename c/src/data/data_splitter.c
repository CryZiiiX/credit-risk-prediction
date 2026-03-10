/*****************************************************************************************************

Nom : src/data/data_splitter.c

Rôle : Division du dataset en ensembles d'entraînement et de test avec mélange aléatoire

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "data_splitter.h"
#include "../utils/memory_manager.h"
#include "../utils/utils.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

/* **************************************************
 * # --- DIVISION TRAIN/TEST --- #
 * ************************************************** */

/**
 * Fonction : shuffle_dataset
 * Rôle     : Mélange aléatoirement les lignes d'un dataset en place
 * Param    : dataset (dataset à mélanger)
 * Retour   : void
 */
void shuffle_dataset(Dataset* dataset) {
    srand(time(NULL));
    
    for (int i = dataset->rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap rows
        double* temp_row = dataset->data[i];
        dataset->data[i] = dataset->data[j];
        dataset->data[j] = temp_row;
        
        // Swap labels
        int temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}

/**
 * Fonction : split_dataset
 * Rôle     : Divise un dataset en ensembles d'entraînement et de test selon un ratio donné
 * Param    : dataset (dataset à diviser), train_ratio (proportion pour l'entraînement, entre 0 et 1)
 * Retour   : SplitData* (structure contenant les datasets train et test)
 */
SplitData* split_dataset(Dataset* dataset, double train_ratio) {
    SplitData* split = (SplitData*)safe_malloc(sizeof(SplitData));
    
    int train_size = (int)(dataset->rows * train_ratio);
    int test_size = dataset->rows - train_size;
    
    // Allocate train dataset
    split->train = (Dataset*)safe_malloc(sizeof(Dataset));
    split->train->rows = train_size;
    split->train->cols = dataset->cols;
    split->train->data = allocate_matrix(train_size, dataset->cols);
    split->train->labels = (int*)safe_malloc(train_size * sizeof(int));
    
    // Allocate test dataset
    split->test = (Dataset*)safe_malloc(sizeof(Dataset));
    split->test->rows = test_size;
    split->test->cols = dataset->cols;
    split->test->data = allocate_matrix(test_size, dataset->cols);
    split->test->labels = (int*)safe_malloc(test_size * sizeof(int));
    
    // Copy data to train
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            split->train->data[i][j] = dataset->data[i][j];
        }
        split->train->labels[i] = dataset->labels[i];
    }
    
    // Copy data to test
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            split->test->data[i][j] = dataset->data[train_size + i][j];
        }
        split->test->labels[i] = dataset->labels[train_size + i];
    }
    
    return split;
}

/**
 * Fonction : split_dataset_from_indices
 * Rôle     : Divise un dataset en ensembles d'entraînement et de test selon un fichier d'indices
 *            Le fichier CSV doit contenir deux colonnes : index,split (0=train, 1=test)
 * Param    : dataset (dataset à diviser), indices_file (chemin vers le fichier CSV d'indices)
 * Retour   : SplitData* (structure contenant les datasets train et test), NULL en cas d'erreur
 */
SplitData* split_dataset_from_indices(Dataset* dataset, const char* indices_file) {
    FILE* file = fopen(indices_file, "r");
    if (!file) {
        fprintf(stderr, "Cannot open indices file: %s\n", indices_file);
        return NULL;
    }
    
    // Lire le header
    char buffer[8192];
    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        fprintf(stderr, "Cannot read header from indices file\n");
        return NULL;
    }
    
    // Allouer un tableau pour stocker les indices de split (0=train, 1=test)
    int* split_indices = (int*)safe_malloc(dataset->rows * sizeof(int));
    int train_count = 0;
    int test_count = 0;
    
    // Lire les indices
    int line_num = 0;
    while (fgets(buffer, sizeof(buffer), file) && line_num < dataset->rows) {
        int index, split;
        if (sscanf(buffer, "%d,%d", &index, &split) == 2) {
            if (index >= 0 && index < dataset->rows) {
                split_indices[index] = split;
                if (split == 0) {
                    train_count++;
                } else if (split == 1) {
                    test_count++;
                }
            }
        }
        line_num++;
    }
    fclose(file);
    
    if (train_count == 0 || test_count == 0) {
        fprintf(stderr, "Invalid split indices: train=%d, test=%d\n", train_count, test_count);
        safe_free(split_indices);
        return NULL;
    }
    
    // Allouer la structure SplitData
    SplitData* split = (SplitData*)safe_malloc(sizeof(SplitData));
    
    // Allocate train dataset
    split->train = (Dataset*)safe_malloc(sizeof(Dataset));
    split->train->rows = train_count;
    split->train->cols = dataset->cols;
    split->train->data = allocate_matrix(train_count, dataset->cols);
    split->train->labels = (int*)safe_malloc(train_count * sizeof(int));
    
    // Allocate test dataset
    split->test = (Dataset*)safe_malloc(sizeof(Dataset));
    split->test->rows = test_count;
    split->test->cols = dataset->cols;
    split->test->data = allocate_matrix(test_count, dataset->cols);
    split->test->labels = (int*)safe_malloc(test_count * sizeof(int));
    
    // Copier les données selon les indices
    int train_idx = 0;
    int test_idx = 0;
    for (int i = 0; i < dataset->rows; i++) {
        if (split_indices[i] == 0) {
            // Train
            for (int j = 0; j < dataset->cols; j++) {
                split->train->data[train_idx][j] = dataset->data[i][j];
            }
            split->train->labels[train_idx] = dataset->labels[i];
            train_idx++;
        } else if (split_indices[i] == 1) {
            // Test
            for (int j = 0; j < dataset->cols; j++) {
                split->test->data[test_idx][j] = dataset->data[i][j];
            }
            split->test->labels[test_idx] = dataset->labels[i];
            test_idx++;
        }
    }
    
    safe_free(split_indices);
    return split;
}

/**
 * Fonction : free_split_data
 * Rôle     : Libère complètement la mémoire allouée pour une structure SplitData
 * Param    : split (structure SplitData à libérer)
 * Retour   : void
 */
void free_split_data(SplitData* split) {
    if (split) {
        free_dataset(split->train);
        free_dataset(split->test);
        safe_free(split);
    }
}

