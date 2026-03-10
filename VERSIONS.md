# Documentation des Versions et Compatibilité

## Vue d'Ensemble

Ce document décrit les versions exactes de tous les composants utilisés dans l'environnement Docker du projet de prédiction du risque de crédit bancaire. L'objectif est de garantir la reproductibilité totale de l'environnement de développement et d'exécution sur n'importe quelle machine compatible Docker.

Le projet utilise un environnement isolé basé sur Docker pour assurer que le code C et les scripts Python d'analyse fonctionnent de manière identique sur toutes les machines, indépendamment des installations système locales.

---

## Environnement Docker

### Image de Base

- **Image** : `gcc:12-bookworm`
- **Registry** : Docker Hub (image officielle)
- **Mainteneur** : Docker Official Images
- **Type** : Image de production stable

**Justification du choix** :
- Image officielle maintenue activement par la communauté Docker
- GCC préinstallé et préconfiguré pour un usage immédiat
- Basée sur Debian Bookworm, distribution stable et bien documentée
- Python 3.11 disponible facilement via les dépôts Debian
- Taille raisonnable (~1.2 GB) pour une image avec compilateur complet
- Excellente compatibilité avec les bibliothèques scientifiques Python

### Système d'Exploitation

- **Distribution** : Debian GNU/Linux 12 (Bookworm)
- **Nom de code** : bookworm
- **Version ID** : 12
- **Architecture** : amd64 (x86_64)
- **Type** : Stable (release officielle)

**Caractéristiques** :
- Support à long terme assuré par la communauté Debian
- Gestion des paquets via APT (Advanced Package Tool)
- Bibliothèques système mises à jour et sécurisées
- Environnement POSIX complet pour le développement C

---

## Composants C

### Compilateur GCC

- **Version** : GCC 12.5.0
- **Date de release** : 2022
- **Organisation** : Free Software Foundation, Inc.
- **Licence** : GPL

**Configuration du projet** :
- Standard C utilisé : C99 (ISO/IEC 9899:1999)
- Flags de compilation : `-Wall -Wextra -O2 -std=c99 -D_POSIX_C_SOURCE=200809L`
- Optimisation : Niveau 2 (O2) pour un bon équilibre performance/taille
- Warnings activés : Tous (-Wall -Wextra) pour garantir la qualité du code
- Bibliothèque mathématique : Liée via `-lm`

**Support** :
- Compilation C99 complète
- Support POSIX 2008
- Extensions GNU disponibles
- Optimisations modernes activées

### Bibliothèques C Standard

- **GNU C Library (glibc)** : Version incluse dans Debian 12
- **libm** : Bibliothèque mathématique standard (fonctions exp, log, sqrt, etc.)
- **Pas de dépendances externes** : Le projet n'utilise que la bibliothèque standard C

### Outils de Build

- **Make** : GNU Make (version incluse dans Debian 12)
- **Binutils** : Collection d'outils binaires GNU pour la compilation et le linkage
- **ld** : GNU linker pour assembler les fichiers objets

---

## Composants Python

### Interpréteur Python

- **Version** : Python 3.11.2
- **Release** : Février 2023
- **Organisation** : Python Software Foundation
- **Installation** : Via les dépôts officiels Debian (apt)

**Caractéristiques de Python 3.11** :
- Support de PEP 668 (gestion des environnements externes)
- Performances améliorées par rapport à Python 3.10
- Nouvelles fonctionnalités de typage
- Meilleure gestion des erreurs et traceback
- Compatibilité avec toutes les bibliothèques scientifiques modernes

### Gestionnaire de Paquets

- **pip** : Version installée via python3-pip (Debian)
- **Mode d'installation** : Avec flag `--break-system-packages` (acceptable dans Docker)
- **Cache** : Désactivé (`--no-cache-dir`) pour réduire la taille de l'image

### Bibliothèques Scientifiques Python

Le projet utilise les bibliothèques suivantes pour l'analyse et la comparaison :

| Bibliothèque | Version Minimale | Rôle dans le Projet | Installation |
|--------------|------------------|---------------------|--------------|
| **pandas** | >= 1.5.0 | Manipulation et chargement des données CSV. Utilisée dans tous les scripts (explore_data.py, plot_results.py, compare_with_sklearn.py, split_dataset.py, analyze_missing_values.py) | Automatique via pip |
| **numpy** | >= 1.23.0 | Calculs numériques, dépendance de pandas, scikit-learn et matplotlib. Utilisée directement dans plot_results.py, compare_with_sklearn.py, explore_data.py, analyze_missing_values.py | Automatique (dépendance) |
| **matplotlib** | >= 3.6.0 | Visualisation des données et résultats (matrices de confusion, courbes ROC, métriques). Utilisée dans plot_results.py et explore_data.py. Inclut mpl_toolkits (make_axes_locatable) | Automatique via pip |
| **seaborn** | >= 0.12.0 | Visualisations statistiques avancées (distributions, corrélations). Utilisée uniquement dans explore_data.py | Automatique via pip |
| **scikit-learn** | >= 1.2.0 | Comparaison des implémentations C avec modèles de référence (LogisticRegression, DecisionTreeClassifier, StandardScaler, LabelEncoder, métriques). Utilisée dans compare_with_sklearn.py et split_dataset.py | Automatique via pip |

**Note** : Les versions installées peuvent être supérieures aux versions minimales spécifiées, selon les dernières versions disponibles sur PyPI au moment du build.

---

## Matrice de Compatibilité

### Compatibilité entre Composants Principaux

| Composant | Version | Compatible avec | Notes |
|-----------|---------|-----------------|-------|
| Debian 12 | bookworm | GCC 12.x, Python 3.11 | Distribution hôte stable |
| GCC 12.5.0 | stable | C99, POSIX 2008 | Compilation sans warnings |
| Python 3.11.2 | stable | pandas 1.5+, numpy 1.23+, scikit-learn 1.2+ | Support natif PEP 668 |
| pandas 1.5.0+ | stable | Python 3.8-3.12, numpy 1.21+ | Manipulation des données |
| scikit-learn 1.2.0+ | stable | Python 3.8-3.12, numpy 1.17+ | Algorithmes ML de référence |

### Dépendances Transitives

Les bibliothèques Python installent automatiquement leurs dépendances :

```
scikit-learn
├── numpy >= 1.17
├── scipy >= 1.3.2
├── joblib >= 1.1.1
└── threadpoolctl >= 2.0.0

pandas
├── numpy >= 1.20.3
├── python-dateutil >= 2.8.1
└── pytz >= 2020.1

seaborn
├── matplotlib >= 3.1
├── numpy >= 1.17
└── pandas >= 0.25

matplotlib
├── numpy >= 1.19
├── pillow >= 6.2.0
├── pyparsing >= 2.2.1
└── python-dateutil >= 2.7
```

---

## Versions Testées et Validées

### Environnements de Test

Le projet a été testé et validé dans les environnements suivants :

1. **Environnement Docker** (recommandé)
   - Image : gcc:12-bookworm
   - Système : Debian 12
   - Architecture : x86_64 (amd64)
   - Statut : Validé et fonctionnel

2. **Environnement Ubuntu 22.04 natif**
   - GCC : 11.x à 13.x
   - Python : 3.10 à 3.11
   - Statut : Compatible (nécessite installation manuelle des dépendances)

3. **Environnement Ubuntu 24.04 natif**
   - GCC : 13.x
   - Python : 3.12
   - Statut : Compatible (nécessite ajustement des versions Python)

### Tests de Validation

- Compilation C : 0 erreurs, 0 warnings avec -Wall -Wextra
- Tests unitaires : 31/31 tests passés
- Exécution du programme : Temps < 2 secondes
- Comparaison avec scikit-learn : Écarts < 1% sur toutes les métriques

---

## Vérification des Versions

### Vérifier les Versions dans le Container Docker

Pour vérifier les versions exactes installées dans votre container :

```bash
# Lancer un shell interactif dans le container
docker run --rm -it credit-risk-predictor:latest bash

# Vérifier GCC
gcc --version
# Sortie attendue : gcc (GCC) 12.5.0

# Vérifier Python
python3 --version
# Sortie attendue : Python 3.11.2

# Vérifier l'OS
cat /etc/os-release
# Sortie attendue : Debian GNU/Linux 12 (bookworm)

# Vérifier Make
make --version
# Sortie attendue : GNU Make 4.3

# Lister les bibliothèques Python installées
pip3 list
# Affiche toutes les bibliothèques avec leurs versions exactes

# Vérifier une bibliothèque spécifique
python3 -c "import sklearn; print(sklearn.__version__)"
python3 -c "import pandas; print(pandas.__version__)"
python3 -c "import numpy; print(numpy.__version__)"
```

### Vérifier depuis l'Extérieur du Container

```bash
# Vérifier la version de l'image de base
docker run --rm gcc:12-bookworm gcc --version

# Vérifier les versions Python après build de votre image
docker run --rm credit-risk-predictor:latest python3 --version
docker run --rm credit-risk-predictor:latest pip3 list
```

---

## Mise à Jour des Dépendances

### Mise à Jour des Bibliothèques Python

Pour mettre à jour les bibliothèques Python vers des versions plus récentes :

1. Modifier `requirements.txt` avec les nouvelles versions souhaitées
2. Rebuild l'image Docker :
   ```bash
   docker build --no-cache -t credit-risk-predictor:latest .
   ```

3. Tester la compatibilité :
   ```bash
   docker run --rm credit-risk-predictor:latest
   docker run --rm credit-risk-predictor:latest bash -c "cd tests && ./run_tests.sh"
   ```

### Mise à Jour de GCC ou Python

Pour changer la version de GCC ou Python, modifier la première ligne du `Dockerfile` :

```dockerfile
# Pour GCC 13
FROM gcc:13-bookworm

# Pour une version spécifique de Python (nécessite une image multi-stage)
FROM python:3.12-slim AS python-base
FROM gcc:12-bookworm
COPY --from=python-base /usr/local /usr/local
```

**Note** : Toute modification majeure nécessite des tests complets de régression.

---

## Notes Techniques

### Particularités de l'Environnement

1. **PEP 668 et --break-system-packages**
   - Python 3.11 dans Debian utilise PEP 668 pour protéger l'environnement système
   - Dans Docker, l'utilisation de `--break-system-packages` est sûre car l'environnement est isolé
   - Aucun risque de corruption de l'installation Python du système hôte

2. **Optimisations GCC**
   - Niveau O2 choisi pour un bon équilibre performance/taille de binaire
   - O3 possible mais peut augmenter la taille du binaire sans gain significatif
   - Pas d'optimisations spécifiques au processeur (-march=native) pour compatibilité maximale

3. **Bibliothèques Scientifiques**
   - numpy et scikit-learn utilisent des bibliothèques natives (BLAS, LAPACK) pour les performances
   - Les versions installées via pip incluent les wheels précompilés optimisés
   - Pas de compilation from source nécessaire

4. **Reproductibilité**
   - Les versions minimales garantissent la compatibilité API
   - Les versions exactes peuvent varier selon la date de build
   - Pour une reproductibilité parfaite, spécifier des versions exactes (==) dans requirements.txt

---

## Compatibilité avec l'Hôte

### Exigences Machine Hôte

Pour exécuter le projet via Docker, la machine hôte doit disposer de :

- **Docker** : Version 20.10 ou supérieure
- **Système d'exploitation** : Linux, macOS, ou Windows avec WSL2
- **Architecture** : x86_64 (amd64)
- **Espace disque** : Minimum 3 GB (2 GB pour l'image + 1 GB workspace)
- **RAM** : Minimum 2 GB disponibles
- **Processeur** : Compatible x86_64 (Intel ou AMD)

### Systèmes Testés

- Ubuntu 22.04 LTS (Jammy Jellyfish)
- Ubuntu 24.04 LTS (Noble Numbat)
- Debian 11 (Bullseye)
- Debian 12 (Bookworm)
- macOS 12+ (avec Docker Desktop)
- Windows 11 avec WSL2

### Limitations Connues

- **ARM/Apple Silicon** : L'image gcc:12-bookworm est x86_64, nécessite l'émulation via Rosetta 2 sur macOS ARM
- **Windows natif** : Docker Desktop avec WSL2 requis, pas de support Docker natif Windows
- **Systèmes 32-bit** : Non supportés

---

## Références

- [Image Docker gcc officielle](https://hub.docker.com/_/gcc)
- [Debian 12 Release Notes](https://www.debian.org/releases/bookworm/)
- [GCC 12 Release Notes](https://gcc.gnu.org/gcc-12/changes.html)
- [Python 3.11 Release Notes](https://docs.python.org/3.11/whatsnew/3.11.html)
- [PEP 668 - Marking Python base environments as "externally managed"](https://peps.python.org/pep-0668/)

---

**Dernière mise à jour** : Décembre 2025

**Version du document** : 1.0

**Compatibilité testée** : Docker 28.2.2, gcc:12-bookworm (décembre 2025)

