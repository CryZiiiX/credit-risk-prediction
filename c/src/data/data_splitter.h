/*****************************************************************************************************

Nom : src/data/data_splitter.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de division train/test

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef DATA_SPLITTER_H
#define DATA_SPLITTER_H

#include "../utils/csv_parser.h"

typedef struct {
    Dataset* train;
    Dataset* test;
} SplitData;

SplitData* split_dataset(Dataset* dataset, double train_ratio);
SplitData* split_dataset_from_indices(Dataset* dataset, const char* indices_file);
void free_split_data(SplitData* split);
void shuffle_dataset(Dataset* dataset);

#endif
