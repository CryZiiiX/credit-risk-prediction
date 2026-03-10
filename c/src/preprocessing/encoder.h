/*****************************************************************************************************

Nom : src/preprocessing/encoder.h

Rôle : Déclarations de fonctions, structures et constantes pour le module d'encodage catégoriel

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef ENCODER_H
#define ENCODER_H

#include "../utils/csv_parser.h"

typedef struct {
    int* mapping;
    int n_classes;
} LabelEncoder;

LabelEncoder* fit_label_encoder(int* labels, int n_samples);
void encode_labels(int* labels, int n_samples, LabelEncoder* encoder);
void free_label_encoder(LabelEncoder* encoder);

// Fonctions d'encodage spécifiques au dataset Credit Risk
int encode_home_ownership(const char* value);
int encode_loan_intent(const char* value);
int encode_loan_grade(const char* value);
int encode_default_on_file(const char* value);

#endif

