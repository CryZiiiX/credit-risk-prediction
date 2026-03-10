/*****************************************************************************************************

Nom : src/preprocessing/encoder.c

Rôle : Encodage des variables catégorielles (home_ownership, loan_grade, etc.)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : N/A

******************************************************************************************************/

#include "encoder.h"
#include "../utils/memory_manager.h"
#include <string.h>

/* **************************************************
 * # --- ENCODAGE CATÉGORIEL --- #
 * ************************************************** */

/**
 * Fonction : fit_label_encoder
 * Rôle     : Crée un encodeur de labels en analysant les valeurs uniques présentes
 * Param    : labels (tableau de labels), n_samples (nombre d'échantillons)
 * Retour   : LabelEncoder* (encodeur configuré)
 */
LabelEncoder* fit_label_encoder(int* labels, int n_samples) {
    LabelEncoder* encoder = (LabelEncoder*)safe_malloc(sizeof(LabelEncoder));
    
    // Find unique labels
    int max_label = labels[0];
    for (int i = 1; i < n_samples; i++) {
        if (labels[i] > max_label) {
            max_label = labels[i];
        }
    }
    
    encoder->n_classes = max_label + 1;
    encoder->mapping = (int*)safe_malloc(encoder->n_classes * sizeof(int));
    
    for (int i = 0; i < encoder->n_classes; i++) {
        encoder->mapping[i] = i;
    }
    
    return encoder;
}

/**
 * Fonction : encode_labels
 * Rôle     : Applique l'encodage de labels à un tableau de labels en place
 * Param    : labels (tableau de labels à encoder), n_samples (nombre d'échantillons), encoder (encodeur à utiliser)
 * Retour   : void
 */
void encode_labels(int* labels, int n_samples, LabelEncoder* encoder) {
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] >= 0 && labels[i] < encoder->n_classes) {
            labels[i] = encoder->mapping[labels[i]];
        }
    }
}

/**
 * Fonction : free_label_encoder
 * Rôle     : Libère complètement la mémoire allouée pour un encodeur de labels
 * Param    : encoder (encodeur à libérer)
 * Retour   : void
 */
void free_label_encoder(LabelEncoder* encoder) {
    if (encoder) {
        safe_free(encoder->mapping);
        safe_free(encoder);
    }
}

/* **************************************************
 * # --- FONCTIONS D'ENCODAGE SPÉCIFIQUES --- #
 * ************************************************** */

/**
 * Fonction : encode_home_ownership
 * Rôle     : Encode la variable catégorielle person_home_ownership en entier
 * Param    : value (chaîne de caractères à encoder : RENT, OWN, MORTGAGE, OTHER)
 * Retour   : int (0=RENT, 1=OWN, 2=MORTGAGE, 3=OTHER)
 */
int encode_home_ownership(const char* value) {
    if (strcmp(value, "RENT") == 0) return 0;
    if (strcmp(value, "OWN") == 0) return 1;
    if (strcmp(value, "MORTGAGE") == 0) return 2;
    if (strcmp(value, "OTHER") == 0) return 3;
    return 0;  // Défaut : RENT
}

/**
 * Fonction : encode_loan_intent
 * Rôle     : Encode la variable catégorielle loan_intent en entier
 * Param    : value (chaîne de caractères à encoder : PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
 * Retour   : int (0=PERSONAL, 1=EDUCATION, 2=MEDICAL, 3=VENTURE, 4=HOMEIMPROVEMENT, 5=DEBTCONSOLIDATION)
 */
int encode_loan_intent(const char* value) {
    if (strcmp(value, "PERSONAL") == 0) return 0;
    if (strcmp(value, "EDUCATION") == 0) return 1;
    if (strcmp(value, "MEDICAL") == 0) return 2;
    if (strcmp(value, "VENTURE") == 0) return 3;
    if (strcmp(value, "HOMEIMPROVEMENT") == 0) return 4;
    if (strcmp(value, "DEBTCONSOLIDATION") == 0) return 5;
    return 0;  // Défaut : PERSONAL
}

/**
 * Fonction : encode_loan_grade
 * Rôle     : Encode la variable ordinale loan_grade en entier
 * Param    : value (chaîne de caractères à encoder : A, B, C, D, E, F, G)
 * Retour   : int (1=A, 2=B, 3=C, 4=D, 5=E, 6=F, 7=G)
 */
int encode_loan_grade(const char* value) {
    if (strcmp(value, "A") == 0) return 1;
    if (strcmp(value, "B") == 0) return 2;
    if (strcmp(value, "C") == 0) return 3;
    if (strcmp(value, "D") == 0) return 4;
    if (strcmp(value, "E") == 0) return 5;
    if (strcmp(value, "F") == 0) return 6;
    if (strcmp(value, "G") == 0) return 7;
    return 3;  // Défaut : C (milieu de gamme)
}

/**
 * Fonction : encode_default_on_file
 * Rôle     : Encode la variable catégorielle cb_person_default_on_file en entier
 * Param    : value (chaîne de caractères à encoder : N, Y)
 * Retour   : int (0=N, 1=Y)
 */
int encode_default_on_file(const char* value) {
    if (strcmp(value, "N") == 0) return 0;
    if (strcmp(value, "Y") == 0) return 1;
    return 0;  // Défaut : N (pas de défaut)
}

