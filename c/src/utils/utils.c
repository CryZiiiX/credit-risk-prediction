/*****************************************************************************************************

Nom : src/utils/utils.c

Rôle : Fonctions utilitaires (vecteurs, matrices, opérations mathématiques)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "utils.h"

/* **************************************************
 * # --- ALLOCATION MÉMOIRE --- #
 * ************************************************** */

/**
 * Fonction : allocate_matrix
 * Rôle     : Alloue dynamiquement une matrice 2D de doubles
 * Param    : rows (nombre de lignes), cols (nombre de colonnes)
 * Retour   : double** (pointeur vers la matrice allouée)
 */
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

/**
 * Fonction : free_matrix
 * Rôle     : Libère la mémoire allouée pour une matrice 2D
 * Param    : matrix (matrice à libérer), rows (nombre de lignes)
 * Retour   : void
 */
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/**
 * Fonction : allocate_vector
 * Rôle     : Alloue dynamiquement un vecteur de doubles
 * Param    : size (taille du vecteur)
 * Retour   : double* (pointeur vers le vecteur alloué)
 */
double* allocate_vector(int size) {
    return (double*)malloc(size * sizeof(double));
}

/**
 * Fonction : free_vector
 * Rôle     : Libère la mémoire allouée pour un vecteur
 * Param    : vector (vecteur à libérer)
 * Retour   : void
 */
void free_vector(double* vector) {
    free(vector);
}

/* **************************************************
 * # --- AFFICHAGE --- #
 * ************************************************** */

/**
 * Fonction : print_matrix
 * Rôle     : Affiche une matrice 2D de doubles sur la sortie standard
 * Param    : matrix (matrice à afficher), rows (nombre de lignes), cols (nombre de colonnes)
 * Retour   : void
 */
void print_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", matrix[i][j]);
        }
        printf("\n");
    }
}

/**
 * Fonction : print_vector
 * Rôle     : Affiche un vecteur de doubles sur la sortie standard
 * Param    : vector (vecteur à afficher), size (taille du vecteur)
 * Retour   : void
 */
void print_vector(double* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.4f ", vector[i]);
    }
    printf("\n");
}

/* **************************************************
 * # --- UTILITAIRES FICHIERS --- #
 * ************************************************** */

/**
 * Fonction : count_lines
 * Rôle     : Compte le nombre de lignes dans un fichier CSV en excluant l'en-tête
 * Param    : filename (nom du fichier à analyser)
 * Retour   : int (nombre de lignes, -1 en cas d'erreur)
 */
int count_lines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;
    
    int count = 0;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }
    fclose(file);
    return count - 1; // Exclude header
}

/**
 * Fonction : count_columns
 * Rôle     : Compte le nombre de colonnes dans la première ligne d'un fichier CSV
 * Param    : filename (nom du fichier à analyser)
 * Retour   : int (nombre de colonnes, -1 en cas d'erreur)
 */
int count_columns(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;
    
    char buffer[4096];
    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        return -1;
    }
    
    int count = 1;
    for (char* p = buffer; *p; p++) {
        if (*p == ',') count++;
    }
    fclose(file);
    return count;
}

