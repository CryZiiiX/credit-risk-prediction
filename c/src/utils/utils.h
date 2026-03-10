/*****************************************************************************************************

Nom : src/utils/utils.h

Rôle : Déclarations de fonctions, structures et constantes pour le module d'utilitaires

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./build/credit_risk_predictor

******************************************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double** allocate_matrix(int rows, int cols);
void free_matrix(double** matrix, int rows);
double* allocate_vector(int size);
void free_vector(double* vector);
void print_matrix(double** matrix, int rows, int cols);
void print_vector(double* vector, int size);
int count_lines(const char* filename);
int count_columns(const char* filename);

#endif
