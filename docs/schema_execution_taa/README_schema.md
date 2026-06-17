# Schema d'execution des scripts - Projet TAA

Schema academique et reutilisable du pipeline d'execution du projet
**Techniques d'apprentissage artificiel** (prediction du risque de credit).
Concu pour etre integre dans le rapport LaTeX.

## But

Montrer, d'un seul coup d'oeil, comment les scripts Python du projet s'enchainent :
de la lecture des donnees brutes jusqu'a l'integration des resultats dans le rapport,
en distinguant l'ordre d'execution du flux de donnees.

## Fichiers analyses pour construire le schema

`main.py` (orchestrateur), `src/data_loader.py`, `src/preprocessing.py`,
`src/split_data.py`, `src/metrics.py`, `src/visualization.py`,
`src/models/{logistic_regression,decision_tree,knn}.py`, ainsi que les dossiers
reels `data/`, `results/`, `reports/figures/` et `reports/latex/`.

## Logique du pipeline (telle qu'executee par main.py)

- **Phase 0 - Preparation des donnees** : `data_loader.load_raw_data()` lit
  `data/raw/credit_risk_dataset.csv`, puis `preprocessing.clean_outliers()`.
- **Phase 1 - Pretraitement et decoupage** : `encode_categoricals()`,
  `split_data.stratified_split()` 70/15/15 (ecrit `data/splits/`), puis
  `MedianImputer` + `StandardScaler` (ajustes sur le train uniquement).
- **Phase 2 - Entrainement des modeles** : regression logistique, arbre CART et
  k-NN, implementes from scratch ; hyperparametres choisis sur la validation.
- **Phase 3 - Evaluation, metriques et figures** : `metrics.py` (accuracy,
  precision, rappel, F1, AUC, matrice de confusion) et `visualization.py`
  (7 figures) ; ecriture de `results/` (metrics.json, predictions.csv,
  models/*.pkl, sklearn_comparison.json) et `reports/figures/`.
- **Phase 4 - Integration au rapport** : `reports/latex/main.tex` inclut les
  figures, puis `pdflatex` produit le PDF livre dans `docs/rapport/`.

## Conventions de couleurs (pastel)

| Couleur | Type de bloc |
|---|---|
| bleu | script Python (`main.py`, `data_loader.py`) |
| vert | traitement (nettoyage, encodage, normalisation) |
| violet | modele (regression logistique, arbre CART, k-NN) |
| turquoise | metriques / visualisation |
| rose | resultats / rapport |
| gris | fichiers / dossiers (stockage) |
| jaune | action manuelle (lancement, compilation) |

## Conventions de fleches

- **fleche pleine** = ordre d'execution ;
- **fleche pointillee** = flux de donnees / fichiers generes.

## Regenerer le schema

```bash
python3 docs/schema_execution_taa/generate_schema.py
```

Modifier les structures `BOXES` / `arrow(...)` / `phase_band(...)` dans
`generate_schema.py` puis relancer la commande. La version Mermaid editable
(`schema_execution_taa.mmd`) peut etre modifiee sur https://mermaid.live.

## Fichiers produits

- `schema_execution_taa.png` - image (200 dpi), pour un apercu rapide ;
- `schema_execution_taa.pdf` - vectoriel, a inclure dans le rapport LaTeX ;
- `schema_execution_taa.svg` - vectoriel editable ;
- `schema_execution_taa.mmd` - source Mermaid (alternative editable) ;
- `generate_schema.py` - script source qui genere SVG, PNG et PDF.

## Notes

- Le schema represente le pipeline reel de `main.py`. La comparaison scikit-learn
  (`results/sklearn_comparison.json`) n'est produite qu'avec l'option
  `--compare-sklearn`.
- Les figures du dossier `data/stats/` (analyse exploratoire) ne sont pas
  generees par `main.py` : elles sont anterieures au pipeline et ne figurent donc
  pas dans le schema.
- Dans le rapport, les figures sont incluses automatiquement (`\includegraphics`)
  tandis que les valeurs chiffrees des metriques sont recopiees a la main depuis
  `results/metrics.json`.
- Inclusion LaTeX : `\includegraphics[width=\textwidth]{schema_execution_taa.pdf}`.
