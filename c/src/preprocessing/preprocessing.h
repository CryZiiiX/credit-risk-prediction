/*****************************************************************************************************

Nom : src/preprocessing/preprocessing.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de prétraitement

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "../utils/csv_parser.h"
#include "scaler.h"

void handle_missing_values(Dataset* dataset);
void normalize_features(Dataset* dataset, Scaler* scaler);
Dataset* preprocess_dataset(Dataset* dataset);

#endif
