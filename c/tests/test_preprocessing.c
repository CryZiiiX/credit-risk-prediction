/*****************************************************************************************************

Nom : tests/test_preprocessing.c

Rôle : Tests unitaires pour le module preprocessing

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./test_preprocessing

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/preprocessing/preprocessing.h"
#include "../src/preprocessing/scaler.h"
#include "../src/utils/csv_parser.h"
#include "../src/utils/memory_manager.h"

void test_scaler_fit() {
    printf("Test 1: Calcul des paramètres du scaler... ");
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 4;
    dataset->cols = 2;
    dataset->data = (double**)safe_malloc(4 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(4 * sizeof(int));
    
    for (int i = 0; i < 4; i++) {
        dataset->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    // Colonne 0: [1, 2, 3, 4] -> mean=2.5, std=1.118
    // Colonne 1: [10, 20, 30, 40] -> mean=25, std=11.18
    dataset->data[0][0] = 1.0; dataset->data[0][1] = 10.0;
    dataset->data[1][0] = 2.0; dataset->data[1][1] = 20.0;
    dataset->data[2][0] = 3.0; dataset->data[2][1] = 30.0;
    dataset->data[3][0] = 4.0; dataset->data[3][1] = 40.0;
    
    Scaler* scaler = fit_scaler(dataset);
    
    assert(scaler != NULL);
    assert(scaler->n_features == 2);
    assert(fabs(scaler->mean[0] - 2.5) < 0.01);
    assert(fabs(scaler->mean[1] - 25.0) < 0.01);
    assert(fabs(scaler->std[0] - 1.118) < 0.1);
    assert(fabs(scaler->std[1] - 11.18) < 0.1);
    
    free_scaler(scaler);
    free_dataset(dataset);
    
    printf("[OK] PASSÉ\n");
}

void test_scaler_transform() {
    printf("Test 2: Transformation des données... ");
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 3;
    dataset->cols = 2;
    dataset->data = (double**)safe_malloc(3 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(3 * sizeof(int));
    
    for (int i = 0; i < 3; i++) {
        dataset->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    dataset->data[0][0] = 0.0; dataset->data[0][1] = 0.0;
    dataset->data[1][0] = 1.0; dataset->data[1][1] = 2.0;
    dataset->data[2][0] = 2.0; dataset->data[2][1] = 4.0;
    
    Scaler* scaler = fit_scaler(dataset);
    transform_dataset(dataset, scaler);
    
    // Vérifier que la moyenne est proche de 0
    double mean_col0 = 0.0, mean_col1 = 0.0;
    for (int i = 0; i < 3; i++) {
        mean_col0 += dataset->data[i][0];
        mean_col1 += dataset->data[i][1];
    }
    mean_col0 /= 3.0;
    mean_col1 /= 3.0;
    
    assert(fabs(mean_col0) < 1e-10);
    assert(fabs(mean_col1) < 1e-10);
    
    free_scaler(scaler);
    free_dataset(dataset);
    
    printf("[OK] PASSÉ\n");
}

void test_scaler_save_load() {
    printf("Test 3: Sauvegarde et chargement du scaler... ");
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 2;
    dataset->cols = 3;
    dataset->data = (double**)safe_malloc(2 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(2 * sizeof(int));
    
    for (int i = 0; i < 2; i++) {
        dataset->data[i] = (double*)safe_malloc(3 * sizeof(double));
    }
    
    dataset->data[0][0] = 1.0; dataset->data[0][1] = 2.0; dataset->data[0][2] = 3.0;
    dataset->data[1][0] = 4.0; dataset->data[1][1] = 5.0; dataset->data[1][2] = 6.0;
    
    Scaler* scaler = fit_scaler(dataset);
    save_scaler("test_scaler.txt", scaler);
    
    Scaler* loaded = load_scaler("test_scaler.txt");
    
    assert(loaded != NULL);
    assert(loaded->n_features == 3);
    
    for (int i = 0; i < 3; i++) {
        assert(fabs(loaded->mean[i] - scaler->mean[i]) < 1e-6);
        assert(fabs(loaded->std[i] - scaler->std[i]) < 1e-6);
    }
    
    free_scaler(scaler);
    free_scaler(loaded);
    free_dataset(dataset);
    remove("test_scaler.txt");
    
    printf("[OK] PASSÉ\n");
}

void test_handle_missing_values() {
    printf("Test 4: Gestion des valeurs manquantes... ");
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 4;
    dataset->cols = 2;
    dataset->data = (double**)safe_malloc(4 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(4 * sizeof(int));
    
    for (int i = 0; i < 4; i++) {
        dataset->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    // Colonne 0: [1, NaN, 3, NaN]
    // Colonne 1: [10, 20, NaN, 40]
    dataset->data[0][0] = 1.0;  dataset->data[0][1] = 10.0;
    dataset->data[1][0] = NAN;  dataset->data[1][1] = 20.0;
    dataset->data[2][0] = 3.0;  dataset->data[2][1] = NAN;
    dataset->data[3][0] = NAN;  dataset->data[3][1] = 40.0;
    
    handle_missing_values(dataset);
    
    // Vérifier qu'il n'y a plus de NaN
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            assert(!isnan(dataset->data[i][j]));
        }
    }
    
    // Les NaN de la colonne 0 devraient être remplacés par mean([1, 3]) = 2.0
    assert(fabs(dataset->data[1][0] - 2.0) < 0.01);
    assert(fabs(dataset->data[3][0] - 2.0) < 0.01);
    
    // Les NaN de la colonne 1 devraient être remplacés par mean([10, 20, 40])
    double expected_mean = (10.0 + 20.0 + 40.0) / 3.0;
    assert(fabs(dataset->data[2][1] - expected_mean) < 0.01);
    
    free_dataset(dataset);
    
    printf("[OK] PASSÉ\n");
}

void test_preprocess_dataset() {
    printf("Test 5: Pipeline de prétraitement complet... ");
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 3;
    dataset->cols = 2;
    dataset->data = (double**)safe_malloc(3 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(3 * sizeof(int));
    
    for (int i = 0; i < 3; i++) {
        dataset->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    dataset->data[0][0] = 1.0;  dataset->data[0][1] = NAN;
    dataset->data[1][0] = 2.0;  dataset->data[1][1] = 5.0;
    dataset->data[2][0] = 3.0;  dataset->data[2][1] = 10.0;
    
    Dataset* result = preprocess_dataset(dataset);
    
    assert(result != NULL);
    // Vérifier qu'il n'y a plus de NaN
    assert(!isnan(result->data[0][1]));
    
    free_dataset(dataset);
    
    printf("[OK] PASSÉ\n");
}

int main() {
    printf("\n=== TESTS DU PRÉTRAITEMENT ===\n\n");
    
    test_scaler_fit();
    test_scaler_transform();
    test_scaler_save_load();
    test_handle_missing_values();
    test_preprocess_dataset();
    
    printf("\n[OK] Tous les tests sont passés avec succès!\n\n");
    return 0;
}

