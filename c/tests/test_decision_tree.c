/*****************************************************************************************************

Nom : tests/test_decision_tree.c

Rôle : Tests unitaires pour le module decision_tree

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./test_decision_tree

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/models/decision_tree.h"
#include "../src/utils/memory_manager.h"

void test_gini_calculation() {
    printf("Test 1: Calcul de Gini... ");
    
    int pure_labels[] = {1, 1, 1, 1};
    int mixed_labels[] = {0, 1, 0, 1};
    int imbalanced_labels[] = {0, 0, 0, 1};
    
    DecisionTree* tree = create_decision_tree(5, 2, 1, GINI);
    
    Dataset* pure_data = (Dataset*)safe_malloc(sizeof(Dataset));
    pure_data->rows = 4;
    pure_data->cols = 1;
    pure_data->data = (double**)safe_malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++) {
        pure_data->data[i] = (double*)safe_malloc(sizeof(double));
        pure_data->data[i][0] = i;
    }
    pure_data->labels = pure_labels;
    
    train_decision_tree(tree, pure_data);
    assert(tree->root->is_leaf == 1);
    assert(fabs(tree->root->impurity - 0.0) < 0.01);
    
    Dataset* mixed_data = (Dataset*)safe_malloc(sizeof(Dataset));
    mixed_data->rows = 4;
    mixed_data->cols = 1;
    mixed_data->data = (double**)safe_malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++) {
        mixed_data->data[i] = (double*)safe_malloc(sizeof(double));
        mixed_data->data[i][0] = i;
    }
    mixed_data->labels = mixed_labels;
    
    DecisionTree* tree2 = create_decision_tree(0, 2, 1, GINI);
    train_decision_tree(tree2, mixed_data);
    assert(fabs(tree2->root->impurity - 0.5) < 0.01);
    
    for (int i = 0; i < 4; i++) {
        safe_free(pure_data->data[i]);
        safe_free(mixed_data->data[i]);
    }
    safe_free(pure_data->data);
    safe_free(mixed_data->data);
    safe_free(pure_data);
    safe_free(mixed_data);
    free_decision_tree(tree);
    free_decision_tree(tree2);
    
    printf("PASSE\n");
}

void test_entropy_calculation() {
    printf("Test 2: Calcul d'Entropie... ");
    
    int pure_labels[] = {0, 0, 0, 0};
    int mixed_labels[] = {0, 1, 0, 1};
    
    DecisionTree* tree = create_decision_tree(5, 2, 1, ENTROPY);
    
    Dataset* pure_data = (Dataset*)safe_malloc(sizeof(Dataset));
    pure_data->rows = 4;
    pure_data->cols = 1;
    pure_data->data = (double**)safe_malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++) {
        pure_data->data[i] = (double*)safe_malloc(sizeof(double));
        pure_data->data[i][0] = i;
    }
    pure_data->labels = pure_labels;
    
    train_decision_tree(tree, pure_data);
    assert(fabs(tree->root->impurity - 0.0) < 0.01);
    
    Dataset* mixed_data = (Dataset*)safe_malloc(sizeof(Dataset));
    mixed_data->rows = 4;
    mixed_data->cols = 1;
    mixed_data->data = (double**)safe_malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++) {
        mixed_data->data[i] = (double*)safe_malloc(sizeof(double));
        mixed_data->data[i][0] = i;
    }
    mixed_data->labels = mixed_labels;
    
    DecisionTree* tree2 = create_decision_tree(0, 2, 1, ENTROPY);
    train_decision_tree(tree2, mixed_data);
    assert(fabs(tree2->root->impurity - 1.0) < 0.01);
    
    for (int i = 0; i < 4; i++) {
        safe_free(pure_data->data[i]);
        safe_free(mixed_data->data[i]);
    }
    safe_free(pure_data->data);
    safe_free(mixed_data->data);
    safe_free(pure_data);
    safe_free(mixed_data);
    free_decision_tree(tree);
    free_decision_tree(tree2);
    
    printf("PASSE\n");
}

void test_simple_tree() {
    printf("Test 3: Arbre simple profondeur 2... ");
    
    double x0[] = {1.0}; int y0 = 0;
    double x1[] = {2.0}; int y1 = 0;
    double x2[] = {5.0}; int y2 = 1;
    double x3[] = {6.0}; int y3 = 1;
    
    Dataset* data = (Dataset*)safe_malloc(sizeof(Dataset));
    data->rows = 4;
    data->cols = 1;
    data->data = (double**)safe_malloc(4 * sizeof(double*));
    data->data[0] = x0;
    data->data[1] = x1;
    data->data[2] = x2;
    data->data[3] = x3;
    data->labels = (int*)safe_malloc(4 * sizeof(int));
    data->labels[0] = y0;
    data->labels[1] = y1;
    data->labels[2] = y2;
    data->labels[3] = y3;
    
    DecisionTree* tree = create_decision_tree(2, 1, 1, GINI);
    train_decision_tree(tree, data);
    
    assert(get_tree_depth(tree) <= 2);
    assert(count_tree_nodes(tree) >= 1);
    
    double test1[] = {1.5};
    double test2[] = {5.5};
    int pred1 = predict_tree_single(tree, test1);
    int pred2 = predict_tree_single(tree, test2);
    
    assert(pred1 == 0);
    assert(pred2 == 1);
    
    safe_free(data->labels);
    safe_free(data->data);
    safe_free(data);
    free_decision_tree(tree);
    
    printf("PASSE\n");
}

void test_predictions() {
    printf("Test 4: Predictions sur dataset... ");
    
    Dataset* data = (Dataset*)safe_malloc(sizeof(Dataset));
    data->rows = 6;
    data->cols = 2;
    data->data = (double**)safe_malloc(6 * sizeof(double*));
    data->labels = (int*)safe_malloc(6 * sizeof(int));
    
    for (int i = 0; i < 6; i++) {
        data->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    data->data[0][0] = 1.0; data->data[0][1] = 2.0; data->labels[0] = 0;
    data->data[1][0] = 1.5; data->data[1][1] = 1.8; data->labels[1] = 0;
    data->data[2][0] = 2.0; data->data[2][1] = 2.5; data->labels[2] = 0;
    data->data[3][0] = 5.0; data->data[3][1] = 6.0; data->labels[3] = 1;
    data->data[4][0] = 5.5; data->data[4][1] = 5.8; data->labels[4] = 1;
    data->data[5][0] = 6.0; data->data[5][1] = 6.2; data->labels[5] = 1;
    
    DecisionTree* tree = create_decision_tree(3, 2, 1, GINI);
    train_decision_tree(tree, data);
    
    int* predictions = predict_tree_dataset(tree, data);
    
    int correct = 0;
    for (int i = 0; i < 6; i++) {
        if (predictions[i] == data->labels[i]) correct++;
    }
    
    assert(correct >= 4);
    
    double* probas = get_tree_probabilities(tree, data);
    for (int i = 0; i < 6; i++) {
        assert(probas[i] >= 0.0 && probas[i] <= 1.0);
    }
    
    safe_free(probas);
    safe_free(predictions);
    for (int i = 0; i < 6; i++) {
        safe_free(data->data[i]);
    }
    safe_free(data->data);
    safe_free(data->labels);
    safe_free(data);
    free_decision_tree(tree);
    
    printf("PASSE\n");
}

void test_max_depth_limit() {
    printf("Test 5: Limite de profondeur... ");
    
    Dataset* data = (Dataset*)safe_malloc(sizeof(Dataset));
    data->rows = 8;
    data->cols = 1;
    data->data = (double**)safe_malloc(8 * sizeof(double*));
    data->labels = (int*)safe_malloc(8 * sizeof(int));
    
    for (int i = 0; i < 8; i++) {
        data->data[i] = (double*)safe_malloc(sizeof(double));
        data->data[i][0] = i;
        data->labels[i] = i % 2;
    }
    
    DecisionTree* tree = create_decision_tree(2, 1, 1, GINI);
    train_decision_tree(tree, data);
    
    int depth = get_tree_depth(tree);
    assert(depth <= 2);
    
    for (int i = 0; i < 8; i++) {
        safe_free(data->data[i]);
    }
    safe_free(data->data);
    safe_free(data->labels);
    safe_free(data);
    free_decision_tree(tree);
    
    printf("PASSE\n");
}

void test_min_samples_split() {
    printf("Test 6: Min samples split... ");
    
    Dataset* data = (Dataset*)safe_malloc(sizeof(Dataset));
    data->rows = 4;
    data->cols = 1;
    data->data = (double**)safe_malloc(4 * sizeof(double*));
    data->labels = (int*)safe_malloc(4 * sizeof(int));
    
    for (int i = 0; i < 4; i++) {
        data->data[i] = (double*)safe_malloc(sizeof(double));
        data->data[i][0] = i;
        data->labels[i] = i % 2;
    }
    
    DecisionTree* tree = create_decision_tree(10, 10, 1, GINI);
    train_decision_tree(tree, data);
    
    assert(tree->root->is_leaf == 1);
    
    for (int i = 0; i < 4; i++) {
        safe_free(data->data[i]);
    }
    safe_free(data->data);
    safe_free(data->labels);
    safe_free(data);
    free_decision_tree(tree);
    
    printf("PASSE\n");
}

void test_save_load() {
    printf("Test 7: Sauvegarde et chargement... ");
    
    Dataset* data = (Dataset*)safe_malloc(sizeof(Dataset));
    data->rows = 6;
    data->cols = 2;
    data->data = (double**)safe_malloc(6 * sizeof(double*));
    data->labels = (int*)safe_malloc(6 * sizeof(int));
    
    for (int i = 0; i < 6; i++) {
        data->data[i] = (double*)safe_malloc(2 * sizeof(double));
    }
    
    data->data[0][0] = 1.0; data->data[0][1] = 2.0; data->labels[0] = 0;
    data->data[1][0] = 1.5; data->data[1][1] = 1.8; data->labels[1] = 0;
    data->data[2][0] = 2.0; data->data[2][1] = 2.5; data->labels[2] = 0;
    data->data[3][0] = 5.0; data->data[3][1] = 6.0; data->labels[3] = 1;
    data->data[4][0] = 5.5; data->data[4][1] = 5.8; data->labels[4] = 1;
    data->data[5][0] = 6.0; data->data[5][1] = 6.2; data->labels[5] = 1;
    
    DecisionTree* tree = create_decision_tree(3, 2, 1, GINI);
    train_decision_tree(tree, data);
    
    int* pred_before = predict_tree_dataset(tree, data);
    
    save_decision_tree("tests/test_tree.bin", tree);
    
    DecisionTree* loaded = load_decision_tree("tests/test_tree.bin");
    assert(loaded != NULL);
    assert(loaded->max_depth == tree->max_depth);
    assert(loaded->criterion == tree->criterion);
    
    int* pred_after = predict_tree_dataset(loaded, data);
    
    for (int i = 0; i < 6; i++) {
        assert(pred_before[i] == pred_after[i]);
    }
    
    safe_free(pred_before);
    safe_free(pred_after);
    
    for (int i = 0; i < 6; i++) {
        safe_free(data->data[i]);
    }
    safe_free(data->data);
    safe_free(data->labels);
    safe_free(data);
    free_decision_tree(tree);
    free_decision_tree(loaded);
    
    printf("PASSE\n");
}

int main() {
    printf("\n=== TESTS DECISION TREE ===\n\n");
    
    test_gini_calculation();
    test_entropy_calculation();
    test_simple_tree();
    test_predictions();
    test_max_depth_limit();
    test_min_samples_split();
    test_save_load();
    
    printf("\nTous les tests sont passes avec succes!\n\n");
    return 0;
}

