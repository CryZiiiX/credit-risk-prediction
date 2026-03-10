/*****************************************************************************************************

Nom : src/models/decision_tree.h

Rôle : Déclarations de fonctions, structures et constantes pour le module d'arbre de décision

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "../utils/csv_parser.h"

typedef enum {
    GINI,
    ENTROPY
} SplitCriterion;

typedef struct DecisionNode {
    int is_leaf;
    int predicted_class;
    double class_probability;
    int feature_index;
    double threshold;
    int n_samples;
    double impurity;
    struct DecisionNode* left;
    struct DecisionNode* right;
} DecisionNode;

typedef struct {
    DecisionNode* root;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    SplitCriterion criterion;
    int n_features;
} DecisionTree;

DecisionTree* create_decision_tree(int max_depth, int min_samples_split, int min_samples_leaf, SplitCriterion criterion);
void train_decision_tree(DecisionTree* tree, Dataset* dataset);
int predict_tree_single(DecisionTree* tree, double* sample);
int* predict_tree_dataset(DecisionTree* tree, Dataset* dataset);
double* get_tree_probabilities(DecisionTree* tree, Dataset* dataset);
int get_tree_depth(DecisionTree* tree);
int count_tree_nodes(DecisionTree* tree);
void print_tree(DecisionTree* tree);
void save_decision_tree(const char* filename, DecisionTree* tree);
DecisionTree* load_decision_tree(const char* filename);
void free_decision_tree(DecisionTree* tree);

#endif

