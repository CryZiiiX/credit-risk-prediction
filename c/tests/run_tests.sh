#!/bin/bash
#/*****************************************************************************************************
#
#Nom : tests/run_tests.sh
#
#Rôle : Script d'exécution des tests unitaires (compilation et exécution de tous les tests)
#
#Auteur : Maxime BRONNY
#
#Version : V1
#
#Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA
#
#Usage :
#
#    Pour compiler : N/A (script shell)
#
#    Pour executer : bash tests/run_tests.sh ou ./tests/run_tests.sh
#
#******************************************************************************************************/

echo "========================================"
echo "   SUITE DE TESTS CREDIT RISK PROJET"
echo "========================================"
echo ""

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Flags de compilation
CC=gcc
CFLAGS="-Wall -Wextra -O2 -std=c99 -D_POSIX_C_SOURCE=200809L"
LDFLAGS="-lm"

# Répertoires
SRC_DIR="../src"
BUILD_DIR="../build/tests"
TESTS_DIR="."

# Créer le répertoire de build pour les tests
mkdir -p $BUILD_DIR

# Fonction pour compiler et exécuter un test
run_test() {
    test_name=$1
    test_file="${TESTS_DIR}/${test_name}.c"
    test_bin="${BUILD_DIR}/${test_name}"
    
    echo -e "${YELLOW}Compilation de ${test_name}...${NC}"
    
    # Compiler le test avec tous les fichiers sources nécessaires
    $CC $CFLAGS -o $test_bin $test_file \
        $SRC_DIR/utils/*.c \
        $SRC_DIR/data/*.c \
        $SRC_DIR/preprocessing/*.c \
        $SRC_DIR/models/*.c \
        $SRC_DIR/evaluation/*.c \
        $LDFLAGS 2>&1
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Échec de compilation de ${test_name}${NC}"
        return 1
    fi
    
    # Exécuter le test
    $test_bin
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] ${test_name} réussi${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${test_name} échoué${NC}"
        echo ""
        return 1
    fi
}

# Compteurs
total=0
passed=0
failed=0

# Liste des tests à exécuter
tests=(
    "test_data_loader"
    "test_preprocessing"
    "test_metrics"
    "test_logistic_regression"
)

# Exécuter chaque test
for test in "${tests[@]}"; do
    total=$((total + 1))
    run_test $test
    if [ $? -eq 0 ]; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
done

# Résumé final
echo "========================================"
echo "            RÉSUMÉ DES TESTS"
echo "========================================"
echo -e "Total:  $total tests"
echo -e "${GREEN}Réussis: $passed tests${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}Échoués: $failed tests${NC}"
else
    echo -e "Échoués: $failed tests"
fi
echo ""

# Code de sortie
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}[OK] Tous les tests sont passés!${NC}"
    exit 0
else
    echo -e "${RED}✗ Certains tests ont échoué${NC}"
    exit 1
fi

