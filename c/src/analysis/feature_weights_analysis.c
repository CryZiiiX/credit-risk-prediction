/*****************************************************************************************************

Nom : src/analysis/feature_weights_analysis.c

Rôle : Module d'analyse des poids des features de la régression logistique

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#include "feature_weights_analysis.h"
#include "../models/logistic_regression.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* **************************************************
 * # --- CONSTANTES --- #
 * ************************************************** */

#define NUM_FEATURES 11

// Noms des features dans l'ordre après prétraitement
static const char* FEATURE_NAMES[NUM_FEATURES] = {
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
};

/* **************************************************
 * # --- FONCTIONS UTILITAIRES --- #
 * ************************************************** */

/**
 * Fonction : compare_abs_weights
 * Rôle     : Fonction de comparaison pour trier par valeur absolue décroissante
 * Param    : a, b (pointeurs vers FeatureWeight)
 * Retour   : int (négatif si a > b, positif si a < b, 0 si égal)
 */
int compare_abs_weights(const void* a, const void* b) {
    FeatureWeight* fw_a = (FeatureWeight*)a;
    FeatureWeight* fw_b = (FeatureWeight*)b;
    
    // Tri décroissant par valeur absolue
    if (fw_a->abs_weight > fw_b->abs_weight) return -1;
    if (fw_a->abs_weight < fw_b->abs_weight) return 1;
    return 0;
}

/**
 * Fonction : generate_interpretation
 * Rôle     : Génère une interprétation automatique basée sur le poids et la feature
 * Param    : feature_name (nom de la feature), weight (poids)
 * Retour   : char* (chaîne d'interprétation allouée dynamiquement)
 */
char* generate_interpretation(const char* feature_name, double weight) {
    char* interpretation = (char*)malloc(256 * sizeof(char));
    if (!interpretation) {
        return NULL;
    }
    
    double abs_w = fabs(weight);
    const char* direction = (weight > 0) ? "↑ Risque de défaut" : "↓ Risque de défaut";
    const char* intensity;
    
    // Déterminer l'intensité basée sur la valeur absolue
    if (abs_w >= 1.0) {
        intensity = "forte influence";
    } else if (abs_w >= 0.5) {
        intensity = "influence modérée";
    } else if (abs_w >= 0.2) {
        intensity = "effet modéré";
    } else {
        intensity = "effet faible";
    }
    
    // Générer l'interprétation selon le type de feature
    if (strcmp(feature_name, "loan_int_rate") == 0) {
        snprintf(interpretation, 256, "↑ Taux d'intérêt → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "loan_grade") == 0) {
        snprintf(interpretation, 256, "Grade moins bon (A→G) → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "cb_person_default_on_file") == 0) {
        snprintf(interpretation, 256, "Historique de défaut → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "loan_percent_income") == 0) {
        snprintf(interpretation, 256, "↑ Ratio prêt/revenu → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "person_emp_length") == 0) {
        snprintf(interpretation, 256, "↑ Ancienneté professionnelle → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "person_income") == 0) {
        snprintf(interpretation, 256, "↑ Revenu → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "person_age") == 0) {
        snprintf(interpretation, 256, "↑ Âge → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "loan_amnt") == 0) {
        snprintf(interpretation, 256, "↑ Montant emprunté → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "loan_intent") == 0) {
        snprintf(interpretation, 256, "Effet dépendant du type de prêt (%s)", intensity);
    } else if (strcmp(feature_name, "person_home_ownership") == 0) {
        snprintf(interpretation, 256, "Propriétaire → %s (%s)", direction, intensity);
    } else if (strcmp(feature_name, "cb_person_cred_hist_length") == 0) {
        snprintf(interpretation, 256, "↑ Ancienneté du crédit → %s (%s)", direction, intensity);
    } else {
        // Interprétation générique
        snprintf(interpretation, 256, "↑ %s → %s (%s)", feature_name, direction, intensity);
    }
    
    return interpretation;
}

/**
 * Fonction : save_feature_weights_analysis
 * Rôle     : Analyse et sauvegarde les poids des features dans un fichier texte formaté
 * Param    : model (modèle de régression logistique), filename (chemin du fichier de sortie)
 * Retour   : void
 */
void save_feature_weights_analysis(LogisticRegression* model, const char* filename) {
    if (!model || !filename) {
        fprintf(stderr, "Erreur: Modèle ou nom de fichier invalide\n");
        return;
    }
    
    if (model->n_features != NUM_FEATURES) {
        fprintf(stderr, "Erreur: Nombre de features inattendu (%d au lieu de %d)\n", 
                model->n_features, NUM_FEATURES);
        return;
    }
    
    // Allouer un tableau de FeatureWeight
    FeatureWeight* features = (FeatureWeight*)malloc(NUM_FEATURES * sizeof(FeatureWeight));
    if (!features) {
        fprintf(stderr, "Erreur: Allocation mémoire échouée\n");
        return;
    }
    
    // Remplir le tableau avec les poids
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i].feature_name = (char*)FEATURE_NAMES[i];
        features[i].weight = model->weights[i];
        features[i].abs_weight = fabs(model->weights[i]);
        features[i].rank = 0;  // Sera calculé après le tri
        features[i].interpretation = generate_interpretation(FEATURE_NAMES[i], model->weights[i]);
    }
    
    // Trier par valeur absolue décroissante
    qsort(features, NUM_FEATURES, sizeof(FeatureWeight), compare_abs_weights);
    
    // Assigner les rangs après le tri
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i].rank = i + 1;
    }
    
    // Ouvrir le fichier de sortie
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Erreur: Impossible de créer le fichier %s\n", filename);
        // Libérer les interprétations allouées
        for (int i = 0; i < NUM_FEATURES; i++) {
            free(features[i].interpretation);
        }
        free(features);
        return;
    }
    
    // Écrire l'en-tête
    fprintf(file, "================================================================================\n");
    fprintf(file, "ANALYSE DES POIDS - REGRESSION LOGISTIQUE C\n");
    fprintf(file, "================================================================================\n\n");
    
    fprintf(file, "Tableau des poids et importance :\n\n");
    fprintf(file, "%-25s %-12s %-10s %-6s %s\n", 
            "Feature", "Poids (w)", "|w|", "Rang", "Interprétation");
    fprintf(file, "--------------------------------------------------------------------------------\n");
    
    // Écrire les données
    for (int i = 0; i < NUM_FEATURES; i++) {
        // Format du poids avec signe
        char weight_str[16];
        if (features[i].weight >= 0) {
            snprintf(weight_str, 16, "+%.3f", features[i].weight);
        } else {
            snprintf(weight_str, 16, "%.3f", features[i].weight);
        }
        
        fprintf(file, "%-25s %-12s %-10.3f %-6d %s\n",
                features[i].feature_name,
                weight_str,
                features[i].abs_weight,
                features[i].rank,
                features[i].interpretation);
    }
    
    fprintf(file, "--------------------------------------------------------------------------------\n");
    
    // Ajouter des informations supplémentaires
    fprintf(file, "\nBiais (intercept): %.6f\n", model->bias);
    fprintf(file, "Nombre de features: %d\n", model->n_features);
    fprintf(file, "Learning rate: %.6f\n", model->learning_rate);
    fprintf(file, "Max iterations: %d\n", model->max_iterations);
    
    fclose(file);
    
    // Libérer les interprétations allouées
    for (int i = 0; i < NUM_FEATURES; i++) {
        free(features[i].interpretation);
    }
    free(features);
}

/**
 * Fonction : generate_feature_weights_table
 * Rôle     : Wrapper pour générer l'analyse des poids (appelée depuis main.c)
 * Param    : model (modèle de régression logistique)
 * Retour   : void
 */
void generate_feature_weights_table(LogisticRegression* model) {
    if (!model) {
        fprintf(stderr, "Erreur: Modèle invalide\n");
        return;
    }
    
    // Créer le répertoire si nécessaire
    int ret = system("mkdir -p results/metrics/logistic_regression");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to create directory results/metrics/logistic_regression\n");
    }
    
    // Générer l'analyse
    save_feature_weights_analysis(model, "results/metrics/logistic_regression/lr_c_feature_weights_analysis.txt");
}

/* **************************************************
 * # --- MAIN (SEULEMENT SI COMPILÉ EN STANDALONE) --- #
 * ************************************************** */

#ifdef FEATURE_WEIGHTS_STANDALONE
int main(void) {
    printf("================================================================================\n");
    printf("ANALYSE DES POIDS - REGRESSION LOGISTIQUE C\n");
    printf("================================================================================\n\n");
    
    printf("Chargement du modèle...\n");
    LogisticRegression* model = load_model("results/models/logistic_model.bin");
    if (!model) {
        fprintf(stderr, "Erreur: Impossible de charger le modèle depuis results/models/logistic_model.bin\n");
        return 1;
    }
    printf("[OK] Modèle chargé: %d features\n\n", model->n_features);
    
    generate_feature_weights_table(model);
    printf("[OK] Analyse sauvegardée: results/metrics/logistic_regression/lr_c_feature_weights_analysis.txt\n");
    
    free_logistic_regression(model);
    return 0;
}
#endif

