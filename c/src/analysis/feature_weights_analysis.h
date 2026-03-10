/*****************************************************************************************************

Nom : src/analysis/feature_weights_analysis.h

Rôle : Déclarations pour le module d'analyse des poids des features de la régression logistique

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef FEATURE_WEIGHTS_ANALYSIS_H
#define FEATURE_WEIGHTS_ANALYSIS_H

#include "../models/logistic_regression.h"

/**
 * Structure représentant le poids et l'importance d'une feature
 */
typedef struct {
    char* feature_name;
    double weight;
    double abs_weight;
    int rank;
    char* interpretation;
} FeatureWeight;

/**
 * Fonction : save_feature_weights_analysis
 * Rôle     : Analyse et sauvegarde les poids des features dans un fichier texte formaté
 * Param    : model (modèle de régression logistique), filename (chemin du fichier de sortie)
 * Retour   : void
 */
void save_feature_weights_analysis(LogisticRegression* model, const char* filename);

/**
 * Fonction : generate_feature_weights_table
 * Rôle     : Wrapper pour générer l'analyse des poids (appelée depuis main.c)
 * Param    : model (modèle de régression logistique)
 * Retour   : void
 */
void generate_feature_weights_table(LogisticRegression* model);

#endif

