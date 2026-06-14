# ======================================================================================
# Nom     : src/split_data.py
# Rôle    : Réalise le découpage stratifié des données en ensembles d'entraînement, de validation et de test.
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Réalisé dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : Module importé par main.py lors de l'exécution du pipeline.
# ======================================================================================

"""Découpage du dataset en trois ensembles : train / validation / test.

Répartition choisie : 70 % / 15 % / 15 %.
- 70 % d'entraînement : assez de données (~22 000 lignes) pour des modèles
  stables ;
- 15 % de validation : sert UNIQUEMENT à choisir les hyperparamètres
  (seuil de décision, profondeur d'arbre, k du k-NN) ;
- 15 % de test : utilisé une seule fois, à la toute fin, pour mesurer la
  performance réelle de généralisation.

Le split est STRATIFIÉ : la proportion de défauts (~22 %) est préservée
dans chaque ensemble, ce qui est important avec des classes déséquilibrées
(un split purement aléatoire pourrait donner un test non représentatif).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import TARGET_COLUMN

SPLITS_DIR = Path("data/splits")


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple:
    """Découpe le DataFrame en (train, val, test) en gardant la même proportion de
    défauts dans chaque ensemble (découpage stratifié, reproductible via seed).

    On traite chaque classe séparément, ce qui évite qu'un tirage aléatoire
    produise un ensemble de test peu représentatif sur des classes déséquilibrées.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Les ratios doivent sommer à 1 (reçu {ratios}).")

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    # Stratification : on découpe chaque classe séparément avec les mêmes
    # proportions, puis on réunit les morceaux.
    for label in df[TARGET_COLUMN].unique():
        idx = df.index[df[TARGET_COLUMN] == label].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train = df.loc[np.concatenate(train_idx)].reset_index(drop=True)
    val = df.loc[np.concatenate(val_idx)].reset_index(drop=True)
    test = df.loc[np.concatenate(test_idx)].reset_index(drop=True)
    return train, val, test


def save_splits(train, val, test, out_dir: Path = SPLITS_DIR) -> None:
    """Enregistre les ensembles train, validation et test au format CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
