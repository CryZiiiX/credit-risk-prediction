/*****************************************************************************************************

Nom : src/utils/csv_parser.h

Rôle : Déclarations de fonctions, structures et constantes pour le module de parsing CSV

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <stdio.h>

typedef struct {
    double** data;
    int* labels;
    int rows;
    int cols;
} Dataset;

Dataset* load_csv(const char* filename, int has_header, int label_col);
Dataset* load_csv_numeric(const char* filename, int has_header, int label_col);
void free_dataset(Dataset* dataset);
char** parse_csv_line(char* line, int* count);
void free_parsed_line(char** tokens, int count);

#endif

