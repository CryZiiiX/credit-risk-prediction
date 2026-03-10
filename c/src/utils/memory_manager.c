/*****************************************************************************************************

Nom : src/utils/memory_manager.c

Rôle : Gestion sécurisée de la mémoire (allocation/libération avec vérifications)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "memory_manager.h"
#include <stdio.h>

/* **************************************************
 * # --- GESTION MÉMOIRE SÉCURISÉE --- #
 * ************************************************** */

/**
 * Fonction : safe_malloc
 * Rôle     : Alloue de la mémoire avec vérification d'erreur et arrêt du programme en cas d'échec
 * Param    : size (taille en octets à allouer)
 * Retour   : void* (pointeur vers la mémoire allouée)
 */
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

/**
 * Fonction : safe_calloc
 * Rôle     : Alloue et initialise à zéro de la mémoire avec vérification d'erreur
 * Param    : num (nombre d'éléments), size (taille de chaque élément)
 * Retour   : void* (pointeur vers la mémoire allouée)
 */
void* safe_calloc(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (!ptr && num > 0 && size > 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

/**
 * Fonction : safe_realloc
 * Rôle     : Réalloue de la mémoire avec vérification d'erreur et arrêt du programme en cas d'échec
 * Param    : ptr (pointeur vers la mémoire existante), size (nouvelle taille en octets)
 * Retour   : void* (pointeur vers la mémoire réallouée)
 */
void* safe_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fprintf(stderr, "Memory reallocation failed\n");
        exit(1);
    }
    return new_ptr;
}

/**
 * Fonction : safe_free
 * Rôle     : Libère la mémoire de manière sécurisée en vérifiant que le pointeur n'est pas NULL
 * Param    : ptr (pointeur vers la mémoire à libérer)
 * Retour   : void
 */
void safe_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

