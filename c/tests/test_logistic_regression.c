/*****************************************************************************************************

Nom : tests/test_logistic_regression.c

Rôle : Tests unitaires pour le module logistic_regression

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./test_logistic_regression

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/models/logistic_regression.h"
#include "../src/utils/csv_parser.h"
#include "../src/utils/memory_manager.h"

void test_model_creation() {
    printf("Test 1: Création du modèle... ");
    
    LogisticRegression* model = create_logistic_regression(5, 0.01, 100);
    
    assert(model != NULL);
    assert(model->n_features == 5);
    assert(model->learning_rate == 0.01);
    assert(model->max_iterations == 100);
    assert(model->bias == 0.0);
    
    // Vérifier que les poids sont initialisés à 0
    for (int i = 0; i < 5; i++) {
        assert(model->weights[i] == 0.0);
    }
    
    free_logistic_regression(model);
    
    printf("[OK] PASSÉ\n");
}

void test_training_simple() {
    printf("Test 2: Entraînement sur données linéairement séparables... ");
    
    // Créer un dataset simple: y = 1 si x1 + x2 > 1, sinon 0
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 4;
    dataset->cols = 2;
    dataset->data = (double**)safe_malloc(4 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(4 * sizeof(int));
    
    for (int i = 0; i < 4; i++) {
        dataset->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    // Points clairement séparables
    dataset->data[0][0] = 0.0; dataset->data[0][1] = 0.0; dataset->labels[0] = 0;
    dataset->data[1][0] = 0.0; dataset->data[1][1] = 2.0; dataset->labels[1] = 1;
    dataset->data[2][0] = 2.0; dataset->data[2][1] = 0.0; dataset->labels[2] = 1;
    dataset->data[3][0] = 2.0; dataset->data[3][1] = 2.0; dataset->labels[3] = 1;
    
    LogisticRegression* model = create_logistic_regression(2, 0.1, 500);
    train_logistic_regression(model, dataset);
    
    // Vérifier que le modèle a appris quelque chose
    assert(fabs(model->weights[0]) > 0.01);
    assert(fabs(model->weights[1]) > 0.01);
    
    // Tester les prédictions
    int* predictions = predict(model, dataset);
    
    // Au moins 3 sur 4 doivent être corrects
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        if (predictions[i] == dataset->labels[i]) correct++;
    }
    assert(correct >= 3);
    
    free(predictions);
    free_logistic_regression(model);
    free_dataset(dataset);
    
    printf("[OK] PASSÉ\n");
}

void test_predict_proba() {
    printf("Test 3: Prédiction de probabilités... ");
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    dataset->rows = 2;
    dataset->cols = 2;
    dataset->data = (double**)safe_malloc(2 * sizeof(double*));
    dataset->labels = (int*)safe_malloc(2 * sizeof(int));
    
    for (int i = 0; i < 2; i++) {
        dataset->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    dataset->data[0][0] = 1.0; dataset->data[0][1] = 1.0;
    dataset->data[1][0] = 2.0; dataset->data[1][1] = 2.0;
    dataset->labels[0] = 0;
    dataset->labels[1] = 1;
    
    LogisticRegression* model = create_logistic_regression(2, 0.01, 100);
    model->weights[0] = 1.0;
    model->weights[1] = 1.0;
    model->bias = -2.0;
    
    double* probas = predict_proba(model, dataset);
    
    // Vérifier que les probabilités sont entre 0 et 1
    for (int i = 0; i < 2; i++) {
        assert(probas[i] >= 0.0 && probas[i] <= 1.0);
    }
    
    free(probas);
    free_logistic_regression(model);
    free_dataset(dataset);
    
    printf("[OK] PASSÉ\n");
}

void test_model_save_load() {
    printf("Test 4: Sauvegarde et chargement du modèle... ");
    
    // Créer et entraîner un modèle simple
    LogisticRegression* model = create_logistic_regression(3, 0.01, 100);
    model->weights[0] = 1.5;
    model->weights[1] = 2.5;
    model->weights[2] = 3.5;
    model->bias = 0.5;
    
    // Sauvegarder
    save_model("test_model.bin", model);
    
    // Charger
    LogisticRegression* loaded_model = load_model("test_model.bin");
    
    assert(loaded_model != NULL);
    assert(loaded_model->n_features == 3);
    assert(fabs(loaded_model->bias - 0.5) < 1e-6);
    assert(fabs(loaded_model->weights[0] - 1.5) < 1e-6);
    assert(fabs(loaded_model->weights[1] - 2.5) < 1e-6);
    assert(fabs(loaded_model->weights[2] - 3.5) < 1e-6);
    
    free_logistic_regression(model);
    free_logistic_regression(loaded_model);
    remove("test_model.bin");
    
    printf("[OK] PASSÉ\n");
}

int main() {
    printf("\n=== TESTS DE LA RÉGRESSION LOGISTIQUE ===\n\n");
    
    test_model_creation();
    test_training_simple();
    test_predict_proba();
    test_model_save_load();
    
    printf("\n[OK] Tous les tests sont passés avec succès!\n\n");
    return 0;
}

