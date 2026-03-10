/*****************************************************************************************************

Nom : src/models/decision_tree.c

Rôle : Implémentation from scratch de l'arbre de décision CART (construction, prédiction, sauvegarde)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "decision_tree.h"
#include "../utils/memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int feature_index;
    double threshold;
    double gain;
} BestSplit;

/* **************************************************
 * # --- CALCUL D'IMPURETÉ (GINI/ENTROPY) --- #
 * ************************************************** */

/**
 * Fonction : compute_gini
 * Rôle     : Calcule l'impureté de Gini pour un ensemble de labels
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons)
 * Retour   : double (valeur d'impureté de Gini entre 0 et 1)
 */
static double compute_gini(int* labels, int n_samples) {
    if (n_samples == 0) return 0.0;
    
    int count_class_0 = 0;
    int count_class_1 = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == 0) count_class_0++;
        else count_class_1++;
    }
    
    double p0 = (double)count_class_0 / n_samples;
    double p1 = (double)count_class_1 / n_samples;
    
    return 1.0 - (p0 * p0 + p1 * p1);
}

/**
 * Fonction : compute_entropy
 * Rôle     : Calcule l'entropie pour un ensemble de labels
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons)
 * Retour   : double (valeur d'entropie en bits)
 */
static double compute_entropy(int* labels, int n_samples) {
    if (n_samples == 0) return 0.0;
    
    int count_class_0 = 0;
    int count_class_1 = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == 0) count_class_0++;
        else count_class_1++;
    }
    
    double p0 = (double)count_class_0 / n_samples;
    double p1 = (double)count_class_1 / n_samples;
    
    double entropy = 0.0;
    if (p0 > 0) entropy -= p0 * log2(p0);
    if (p1 > 0) entropy -= p1 * log2(p1);
    
    return entropy;
}

/**
 * Fonction : compute_impurity
 * Rôle     : Calcule l'impureté selon le critère spécifié (Gini ou Entropy)
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons), criterion (critère de division)
 * Retour   : double (valeur d'impureté)
 */
static double compute_impurity(int* labels, int n_samples, SplitCriterion criterion) {
    if (criterion == GINI) {
        return compute_gini(labels, n_samples);
    } else {
        return compute_entropy(labels, n_samples);
    }
}

/**
 * Fonction : majority_class
 * Rôle     : Détermine la classe majoritaire dans un ensemble de labels
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons)
 * Retour   : int (classe majoritaire, 0 ou 1)
 */
static int majority_class(int* labels, int n_samples) {
    int count_0 = 0;
    int count_1 = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == 0) count_0++;
        else count_1++;
    }
    
    return (count_1 >= count_0) ? 1 : 0;
}

/**
 * Fonction : class_probability
 * Rôle     : Calcule la probabilité de la classe positive dans un ensemble de labels
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons)
 * Retour   : double (probabilité entre 0 et 1)
 */
static double class_probability(int* labels, int n_samples) {
    if (n_samples == 0) return 0.0;
    
    int count_1 = 0;
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == 1) count_1++;
    }
    
    return (double)count_1 / n_samples;
}

/* **************************************************
 * # --- CONSTRUCTION DE L'ARBRE --- #
 * ************************************************** */

/**
 * Fonction : create_leaf_node
 * Rôle     : Crée un nœud feuille avec la classe majoritaire et sa probabilité
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons), criterion (critère d'impureté)
 * Retour   : DecisionNode* (nœud feuille créé)
 */
static DecisionNode* create_leaf_node(int* labels, int n_samples, SplitCriterion criterion) {
    DecisionNode* node = (DecisionNode*)safe_malloc(sizeof(DecisionNode));
    node->is_leaf = 1;
    node->predicted_class = majority_class(labels, n_samples);
    node->class_probability = class_probability(labels, n_samples);
    node->feature_index = -1;
    node->threshold = 0.0;
    node->n_samples = n_samples;
    node->impurity = compute_impurity(labels, n_samples, criterion);
    node->left = NULL;
    node->right = NULL;
    return node;
}

/**
 * Fonction : create_internal_node
 * Rôle     : Crée un nœud interne avec un critère de division
 * Param    : feature_idx (index de la feature), threshold (seuil de division), n_samples (nombre d'échantillons), impurity (impureté)
 * Retour   : DecisionNode* (nœud interne créé)
 */
static DecisionNode* create_internal_node(int feature_idx, double threshold, int n_samples, double impurity) {
    DecisionNode* node = (DecisionNode*)safe_malloc(sizeof(DecisionNode));
    node->is_leaf = 0;
    node->predicted_class = -1;
    node->class_probability = 0.0;
    node->feature_index = feature_idx;
    node->threshold = threshold;
    node->n_samples = n_samples;
    node->impurity = impurity;
    node->left = NULL;
    node->right = NULL;
    return node;
}

/**
 * Fonction : find_best_split
 * Rôle     : Trouve la meilleure division en testant toutes les features et tous les seuils possibles
 * Param    : data (matrice de données), labels (tableau de labels), n_samples (nombre d'échantillons), n_features (nombre de features), criterion (critère de division)
 * Retour   : BestSplit (meilleure division trouvée avec gain maximal)
 */
static BestSplit find_best_split(double** data, int* labels, int n_samples, int n_features, SplitCriterion criterion) {
    BestSplit best = {-1, 0.0, -1.0};
    double parent_impurity = compute_impurity(labels, n_samples, criterion);
    
    for (int feature = 0; feature < n_features; feature++) {
        double* values = (double*)safe_malloc(n_samples * sizeof(double));
        int* indices = (int*)safe_malloc(n_samples * sizeof(int));
        
        for (int i = 0; i < n_samples; i++) {
            values[i] = data[i][feature];
            indices[i] = i;
        }
        
        for (int i = 0; i < n_samples - 1; i++) {
            for (int j = i + 1; j < n_samples; j++) {
                if (values[i] > values[j]) {
                    double temp_val = values[i];
                    values[i] = values[j];
                    values[j] = temp_val;
                    
                    int temp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp_idx;
                }
            }
        }
        
        for (int i = 0; i < n_samples - 1; i++) {
            if (values[i] == values[i + 1]) continue;
            
            double threshold = (values[i] + values[i + 1]) / 2.0;
            
            int n_left = i + 1;
            int n_right = n_samples - n_left;
            
            if (n_left == 0 || n_right == 0) continue;
            
            int* left_labels = (int*)safe_malloc(n_left * sizeof(int));
            int* right_labels = (int*)safe_malloc(n_right * sizeof(int));
            
            for (int j = 0; j <= i; j++) {
                left_labels[j] = labels[indices[j]];
            }
            for (int j = i + 1; j < n_samples; j++) {
                right_labels[j - i - 1] = labels[indices[j]];
            }
            
            double left_impurity = compute_impurity(left_labels, n_left, criterion);
            double right_impurity = compute_impurity(right_labels, n_right, criterion);
            
            double weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples;
            double gain = parent_impurity - weighted_impurity;
            
            if (gain > best.gain) {
                best.gain = gain;
                best.feature_index = feature;
                best.threshold = threshold;
            }
            
            safe_free(left_labels);
            safe_free(right_labels);
        }
        
        safe_free(values);
        safe_free(indices);
    }
    
    return best;
}

/**
 * Fonction : build_tree
 * Rôle     : Construit récursivement l'arbre de décision en appliquant les critères d'arrêt
 * Param    : data (matrice de données), labels (tableau de labels), n_samples (nombre d'échantillons), n_features (nombre de features), depth (profondeur actuelle), tree (arbre contenant les hyperparamètres)
 * Retour   : DecisionNode* (racine de l'arbre construit)
 */
static DecisionNode* build_tree(double** data, int* labels, int n_samples, int n_features, 
                                int depth, DecisionTree* tree) {
    double impurity = compute_impurity(labels, n_samples, tree->criterion);
    
    if (depth >= tree->max_depth || n_samples < tree->min_samples_split || impurity == 0.0) {
        return create_leaf_node(labels, n_samples, tree->criterion);
    }
    
    BestSplit best = find_best_split(data, labels, n_samples, n_features, tree->criterion);
    
    if (best.gain <= 0.0) {
        return create_leaf_node(labels, n_samples, tree->criterion);
    }
    
    double** left_data = (double**)safe_malloc(n_samples * sizeof(double*));
    double** right_data = (double**)safe_malloc(n_samples * sizeof(double*));
    int* left_labels = (int*)safe_malloc(n_samples * sizeof(int));
    int* right_labels = (int*)safe_malloc(n_samples * sizeof(int));
    
    int n_left = 0;
    int n_right = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (data[i][best.feature_index] <= best.threshold) {
            left_data[n_left] = data[i];
            left_labels[n_left] = labels[i];
            n_left++;
        } else {
            right_data[n_right] = data[i];
            right_labels[n_right] = labels[i];
            n_right++;
        }
    }
    
    if (n_left < tree->min_samples_leaf || n_right < tree->min_samples_leaf) {
        safe_free(left_data);
        safe_free(right_data);
        safe_free(left_labels);
        safe_free(right_labels);
        return create_leaf_node(labels, n_samples, tree->criterion);
    }
    
    DecisionNode* node = create_internal_node(best.feature_index, best.threshold, n_samples, impurity);
    
    node->left = build_tree(left_data, left_labels, n_left, n_features, depth + 1, tree);
    node->right = build_tree(right_data, right_labels, n_right, n_features, depth + 1, tree);
    
    safe_free(left_data);
    safe_free(right_data);
    safe_free(left_labels);
    safe_free(right_labels);
    
    return node;
}

/* **************************************************
 * # --- PRÉDICTION --- #
 * ************************************************** */

/**
 * Fonction : predict_single
 * Rôle     : Prédit la classe d'un échantillon unique en parcourant l'arbre récursivement
 * Param    : node (nœud racine de l'arbre), sample (vecteur de features de l'échantillon)
 * Retour   : int (classe prédite, 0 ou 1)
 */
static int predict_single(DecisionNode* node, double* sample) {
    if (node->is_leaf) {
        return node->predicted_class;
    }
    
    if (sample[node->feature_index] <= node->threshold) {
        return predict_single(node->left, sample);
    } else {
        return predict_single(node->right, sample);
    }
}

/**
 * Fonction : predict_proba_single
 * Rôle     : Calcule la probabilité de classe positive pour un échantillon unique
 * Param    : node (nœud racine de l'arbre), sample (vecteur de features de l'échantillon)
 * Retour   : double (probabilité entre 0 et 1)
 */
static double predict_proba_single(DecisionNode* node, double* sample) {
    if (node->is_leaf) {
        return node->class_probability;
    }
    
    if (sample[node->feature_index] <= node->threshold) {
        return predict_proba_single(node->left, sample);
    } else {
        return predict_proba_single(node->right, sample);
    }
}

/**
 * Fonction : get_depth_recursive
 * Rôle     : Calcule récursivement la profondeur maximale de l'arbre
 * Param    : node (nœud racine de l'arbre)
 * Retour   : int (profondeur maximale)
 */
static int get_depth_recursive(DecisionNode* node) {
    if (node == NULL || node->is_leaf) {
        return 0;
    }
    
    int left_depth = get_depth_recursive(node->left);
    int right_depth = get_depth_recursive(node->right);
    
    return 1 + (left_depth > right_depth ? left_depth : right_depth);
}

/**
 * Fonction : count_nodes_recursive
 * Rôle     : Compte récursivement le nombre total de nœuds dans l'arbre
 * Param    : node (nœud racine de l'arbre)
 * Retour   : int (nombre total de nœuds)
 */
static int count_nodes_recursive(DecisionNode* node) {
    if (node == NULL) return 0;
    return 1 + count_nodes_recursive(node->left) + count_nodes_recursive(node->right);
}

/* **************************************************
 * # --- AFFICHAGE ET SAUVEGARDE --- #
 * ************************************************** */

/**
 * Fonction : print_tree_recursive
 * Rôle     : Affiche récursivement la structure de l'arbre avec indentation
 * Param    : node (nœud à afficher), depth (profondeur actuelle pour l'indentation)
 * Retour   : void
 */
static void print_tree_recursive(DecisionNode* node, int depth) {
    if (node == NULL) return;
    
    for (int i = 0; i < depth; i++) printf("  ");
    
    if (node->is_leaf) {
        printf("LEAF: class=%d, proba=%.3f, samples=%d, impurity=%.3f\n",
               node->predicted_class, node->class_probability, node->n_samples, node->impurity);
    } else {
        printf("NODE: feature=%d, threshold=%.3f, samples=%d, impurity=%.3f\n",
               node->feature_index, node->threshold, node->n_samples, node->impurity);
        print_tree_recursive(node->left, depth + 1);
        print_tree_recursive(node->right, depth + 1);
    }
}

/**
 * Fonction : save_tree_recursive
 * Rôle     : Sauvegarde récursivement la structure de l'arbre dans un fichier
 * Param    : file (fichier de destination), node (nœud à sauvegarder)
 * Retour   : void
 */
static void save_tree_recursive(FILE* file, DecisionNode* node) {
    if (node == NULL) {
        fprintf(file, "NULL\n");
        return;
    }
    
    if (node->is_leaf) {
        fprintf(file, "LEAF %d %.6f %d %.6f\n",
                node->predicted_class, node->class_probability, node->n_samples, node->impurity);
    } else {
        fprintf(file, "NODE %d %.6f %d %.6f\n",
                node->feature_index, node->threshold, node->n_samples, node->impurity);
        save_tree_recursive(file, node->left);
        save_tree_recursive(file, node->right);
    }
}

/**
 * Fonction : load_tree_recursive
 * Rôle     : Charge récursivement la structure de l'arbre depuis un fichier
 * Param    : file (fichier source)
 * Retour   : DecisionNode* (nœud racine chargé, NULL en cas d'erreur)
 */
static DecisionNode* load_tree_recursive(FILE* file) {
    char type[10];
    if (fscanf(file, "%s", type) != 1) return NULL;
    
    if (strcmp(type, "NULL") == 0) {
        return NULL;
    }
    
    DecisionNode* node = (DecisionNode*)safe_malloc(sizeof(DecisionNode));
    
    if (strcmp(type, "LEAF") == 0) {
        node->is_leaf = 1;
        int items_read = fscanf(file, "%d %lf %d %lf",
                                 &node->predicted_class, &node->class_probability, &node->n_samples, &node->impurity);
        if (items_read != 4) {
            fprintf(stderr, "Error: Failed to read leaf node data\n");
            free(node);
            return NULL;
        }
        node->feature_index = -1;
        node->threshold = 0.0;
        node->left = NULL;
        node->right = NULL;
    } else {
        node->is_leaf = 0;
        int items_read = fscanf(file, "%d %lf %d %lf",
                                 &node->feature_index, &node->threshold, &node->n_samples, &node->impurity);
        if (items_read != 4) {
            fprintf(stderr, "Error: Failed to read internal node data\n");
            free(node);
            return NULL;
        }
        node->predicted_class = -1;
        node->class_probability = 0.0;
        node->left = load_tree_recursive(file);
        node->right = load_tree_recursive(file);
    }
    
    return node;
}

/**
 * Fonction : free_tree_recursive
 * Rôle     : Libère récursivement toute la mémoire allouée pour l'arbre
 * Param    : node (nœud racine à libérer)
 * Retour   : void
 */
static void free_tree_recursive(DecisionNode* node) {
    if (node == NULL) return;
    
    free_tree_recursive(node->left);
    free_tree_recursive(node->right);
    safe_free(node);
}

/* **************************************************
 * # --- FONCTIONS PUBLIQUES --- #
 * ************************************************** */

/**
 * Fonction : create_decision_tree
 * Rôle     : Crée et initialise un arbre de décision avec les hyperparamètres spécifiés
 * Param    : max_depth (profondeur maximale), min_samples_split (minimum d'échantillons pour diviser), min_samples_leaf (minimum d'échantillons par feuille), criterion (critère de division)
 * Retour   : DecisionTree* (arbre initialisé)
 */
DecisionTree* create_decision_tree(int max_depth, int min_samples_split, int min_samples_leaf, SplitCriterion criterion) {
    DecisionTree* tree = (DecisionTree*)safe_malloc(sizeof(DecisionTree));
    tree->root = NULL;
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->criterion = criterion;
    tree->n_features = 0;
    return tree;
}

/**
 * Fonction : train_decision_tree
 * Rôle     : Entraîne l'arbre de décision en construisant récursivement la structure optimale
 * Param    : tree (arbre à entraîner), dataset (dataset d'entraînement)
 * Retour   : void
 */
void train_decision_tree(DecisionTree* tree, Dataset* dataset) {
    tree->n_features = dataset->cols;
    tree->root = build_tree(dataset->data, dataset->labels, dataset->rows, dataset->cols, 0, tree);
}

/**
 * Fonction : predict_tree_single
 * Rôle     : Prédit la classe d'un échantillon unique
 * Param    : tree (arbre entraîné), sample (vecteur de features de l'échantillon)
 * Retour   : int (classe prédite, 0 ou 1)
 */
int predict_tree_single(DecisionTree* tree, double* sample) {
    return predict_single(tree->root, sample);
}

/**
 * Fonction : predict_tree_dataset
 * Rôle     : Prédit les classes pour tous les échantillons d'un dataset
 * Param    : tree (arbre entraîné), dataset (dataset à prédire)
 * Retour   : int* (tableau de prédictions binaires)
 */
int* predict_tree_dataset(DecisionTree* tree, Dataset* dataset) {
    int* predictions = (int*)safe_malloc(dataset->rows * sizeof(int));
    
    for (int i = 0; i < dataset->rows; i++) {
        predictions[i] = predict_single(tree->root, dataset->data[i]);
    }
    
    return predictions;
}

/**
 * Fonction : get_tree_probabilities
 * Rôle     : Calcule les probabilités de classe positive pour tous les échantillons d'un dataset
 * Param    : tree (arbre entraîné), dataset (dataset à prédire)
 * Retour   : double* (tableau de probabilités entre 0 et 1)
 */
double* get_tree_probabilities(DecisionTree* tree, Dataset* dataset) {
    double* probabilities = (double*)safe_malloc(dataset->rows * sizeof(double));
    
    for (int i = 0; i < dataset->rows; i++) {
        probabilities[i] = predict_proba_single(tree->root, dataset->data[i]);
    }
    
    return probabilities;
}

/**
 * Fonction : get_tree_depth
 * Rôle     : Retourne la profondeur maximale de l'arbre
 * Param    : tree (arbre à analyser)
 * Retour   : int (profondeur maximale)
 */
int get_tree_depth(DecisionTree* tree) {
    return get_depth_recursive(tree->root);
}

/**
 * Fonction : count_tree_nodes
 * Rôle     : Retourne le nombre total de nœuds dans l'arbre
 * Param    : tree (arbre à analyser)
 * Retour   : int (nombre total de nœuds)
 */
int count_tree_nodes(DecisionTree* tree) {
    return count_nodes_recursive(tree->root);
}

/**
 * Fonction : print_tree
 * Rôle     : Affiche les informations et la structure complète de l'arbre
 * Param    : tree (arbre à afficher)
 * Retour   : void
 */
void print_tree(DecisionTree* tree) {
    printf("\n=== Decision Tree Structure ===\n");
    printf("Max Depth: %d\n", tree->max_depth);
    printf("Min Samples Split: %d\n", tree->min_samples_split);
    printf("Min Samples Leaf: %d\n", tree->min_samples_leaf);
    printf("Criterion: %s\n", tree->criterion == GINI ? "Gini" : "Entropy");
    printf("Actual Depth: %d\n", get_tree_depth(tree));
    printf("Total Nodes: %d\n", count_tree_nodes(tree));
    printf("\nTree Structure:\n");
    print_tree_recursive(tree->root, 0);
}

/**
 * Fonction : save_decision_tree
 * Rôle     : Sauvegarde un arbre de décision dans un fichier texte
 * Param    : filename (nom du fichier de destination), tree (arbre à sauvegarder)
 * Retour   : void
 */
void save_decision_tree(const char* filename, DecisionTree* tree) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "%d %d %d %d %d\n",
            tree->max_depth, tree->min_samples_split, tree->min_samples_leaf,
            tree->criterion, tree->n_features);
    
    save_tree_recursive(file, tree->root);
    
    fclose(file);
}

/**
 * Fonction : load_decision_tree
 * Rôle     : Charge un arbre de décision depuis un fichier texte
 * Param    : filename (nom du fichier source)
 * Retour   : DecisionTree* (arbre chargé, NULL en cas d'erreur)
 */
DecisionTree* load_decision_tree(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    DecisionTree* tree = (DecisionTree*)safe_malloc(sizeof(DecisionTree));
    int criterion_int;
    
    int items_read = fscanf(file, "%d %d %d %d %d",
                            &tree->max_depth, &tree->min_samples_split, &tree->min_samples_leaf,
                            &criterion_int, &tree->n_features);
    if (items_read != 5) {
        fprintf(stderr, "Error: Failed to read tree parameters\n");
        fclose(file);
        free(tree);
        return NULL;
    }
    
    tree->criterion = (SplitCriterion)criterion_int;
    tree->root = load_tree_recursive(file);
    
    fclose(file);
    return tree;
}

/**
 * Fonction : free_decision_tree
 * Rôle     : Libère complètement la mémoire allouée pour un arbre de décision
 * Param    : tree (arbre à libérer)
 * Retour   : void
 */
void free_decision_tree(DecisionTree* tree) {
    if (tree == NULL) return;
    free_tree_recursive(tree->root);
    safe_free(tree);
}

