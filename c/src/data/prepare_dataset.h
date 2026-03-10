/*****************************************************************************************************

Nom : src/data/prepare_dataset.h

Rôle : Déclarations pour le programme de préparation du dataset (split train/test)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make prepare_dataset

    Pour executer : ./build/prepare_dataset

******************************************************************************************************/

#ifndef PREPARE_DATASET_H
#define PREPARE_DATASET_H

/**
 * Fonction : prepare_dataset
 * Rôle     : Génère le split train/test et sauvegarde les fichiers nécessaires
 * Param    : aucun
 * Retour   : int (0 en cas de succès, 1 en cas d'erreur)
 */
int prepare_dataset(void);

#endif

