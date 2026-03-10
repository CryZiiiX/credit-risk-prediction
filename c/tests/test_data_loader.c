/*****************************************************************************************************

Nom : tests/test_data_loader.c

Rôle : Tests unitaires pour le module data_loader

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage :

    Pour compiler : make

    Pour executer : ./test_data_loader

******************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../src/utils/csv_parser.h"
#include "../src/data/data_loader.h"

void test_load_csv_basic() {
    printf("Test 1: Chargement CSV basique... ");
    
    // Créer un fichier CSV de test
    FILE* f = fopen("test_temp.csv", "w");
    fprintf(f, "a,b,c,label\n");
    fprintf(f, "1.0,2.0,3.0,0\n");
    fprintf(f, "4.0,5.0,6.0,1\n");
    fprintf(f, "7.0,8.0,9.0,0\n");
    fclose(f);
    
    Dataset* dataset = load_csv("test_temp.csv", 1, 3);
    
    assert(dataset != NULL);
    assert(dataset->rows == 3);
    assert(dataset->cols == 3);
    assert(dataset->labels[0] == 0);
    assert(dataset->labels[1] == 1);
    assert(dataset->labels[2] == 0);
    
    free_dataset(dataset);
    remove("test_temp.csv");
    
    printf("[OK] PASSÉ\n");
}

void test_load_csv_without_header() {
    printf("Test 2: Chargement CSV sans header... ");
    
    FILE* f = fopen("test_temp2.csv", "w");
    fprintf(f, "1.0,2.0,0\n");
    fprintf(f, "3.0,4.0,1\n");
    fclose(f);
    
    Dataset* dataset = load_csv("test_temp2.csv", 0, 2);
    
    assert(dataset != NULL);
    assert(dataset->rows == 2);
    assert(dataset->cols == 2);
    
    free_dataset(dataset);
    remove("test_temp2.csv");
    
    printf("[OK] PASSÉ\n");
}

void test_save_dataset() {
    printf("Test 3: Sauvegarde de dataset... ");
    
    // Créer un dataset simple
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->rows = 2;
    dataset->cols = 2;
    dataset->data = (double**)malloc(2 * sizeof(double*));
    dataset->data[0] = (double*)malloc(2 * sizeof(double));
    dataset->data[1] = (double*)malloc(2 * sizeof(double));
    dataset->labels = (int*)malloc(2 * sizeof(int));
    
    dataset->data[0][0] = 1.5;
    dataset->data[0][1] = 2.5;
    dataset->data[1][0] = 3.5;
    dataset->data[1][1] = 4.5;
    dataset->labels[0] = 0;
    dataset->labels[1] = 1;
    
    save_dataset("test_save.csv", dataset);
    
    // Vérifier que le fichier existe
    FILE* f = fopen("test_save.csv", "r");
    assert(f != NULL);
    fclose(f);
    
    free_dataset(dataset);
    remove("test_save.csv");
    
    printf("[OK] PASSÉ\n");
}

void test_categorical_encoding() {
    printf("Test 4: Encodage catégoriel... ");
    
    // Créer un CSV avec des valeurs catégorielles
    FILE* f = fopen("test_cat.csv", "w");
    fprintf(f, "age,income,home,emp,intent,grade,amnt,rate,status,percent,default,hist\n");
    fprintf(f, "25,50000,RENT,5.0,PERSONAL,A,10000,10.0,0,0.2,N,3\n");
    fprintf(f, "30,60000,OWN,10.0,EDUCATION,B,15000,12.0,1,0.25,Y,5\n");
    fclose(f);
    
    Dataset* dataset = load_csv("test_cat.csv", 1, 8);
    
    assert(dataset != NULL);
    assert(dataset->rows == 2);
    
    // Vérifier l'encodage (colonne 2 = home_ownership)
    // RENT=0, OWN=1
    assert(dataset->data[0][2] == 0.0);  // RENT
    assert(dataset->data[1][2] == 1.0);  // OWN
    
    // Colonne 5 dans CSV = loan_grade (A=1, B=2), mais après retrait col 8, c'est col 5
    assert(dataset->data[0][5] == 1.0);  // A
    assert(dataset->data[1][5] == 2.0);  // B
    
    // Colonne 10 dans CSV = default_on_file (N=0, Y=1), après retrait col 8, c'est col 9
    assert(dataset->data[0][9] == 0.0);  // N
    assert(dataset->data[1][9] == 1.0);  // Y
    
    free_dataset(dataset);
    remove("test_cat.csv");
    
    printf("[OK] PASSÉ\n");
}

int main() {
    printf("\n=== TESTS DU DATA LOADER ===\n\n");
    
    test_load_csv_basic();
    test_load_csv_without_header();
    test_save_dataset();
    test_categorical_encoding();
    
    printf("\n[OK] Tous les tests sont passés avec succès!\n\n");
    return 0;
}

