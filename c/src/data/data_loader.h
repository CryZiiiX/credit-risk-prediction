/*****************************************************************************************************

Nom : src/data/data_loader.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de chargement de données

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "../utils/csv_parser.h"

Dataset* load_train_data(const char* filename);
Dataset* load_test_data(const char* filename);
void save_dataset(const char* filename, Dataset* dataset);
void save_dataset_with_header(const char* filename, Dataset* dataset);

#endif
