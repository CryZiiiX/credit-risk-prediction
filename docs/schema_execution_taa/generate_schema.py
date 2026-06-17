#!/usr/bin/env python3
# ======================================================================================
# Nom     : docs/schema_execution_taa/generate_schema.py
# Role    : Genere le schema d'execution des scripts du projet TAA (SVG, PNG, PDF).
# Auteur  : Maxime BRONNY
# Version : V1
# Cadre   : Realise dans le cadre du cours Techniques d'apprentissage artificiel
#           Master 1 Informatique Big Data
# Usage   : python3 docs/schema_execution_taa/generate_schema.py
# ======================================================================================
"""Schema d'execution des scripts du projet TAA (pipeline Python).

Genere un schema academique, pastel et lisible a partir du pipeline reel
(main.py -> src/ -> results/ + reports/figures/ -> reports/latex/).
Sorties : schema_execution_taa.svg / .png / .pdf (dans ce meme dossier).

Le schema est entierement decrit par les structures BOXES / EDGES / PHASES
ci-dessous : il suffit de les modifier puis de relancer ce script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent
W, H = 200.0, 300.0  # repere logique du schema (x : 0-200, y : 0-300)

# --------------------------------------------------------------------------------------
# Palette pastel (remplissage, bordure) par type de bloc
# --------------------------------------------------------------------------------------
COLORS = {
    "script":  ("#BCD4EE", "#5E82AC"),   # bleu pastel      - scripts Python
    "process": ("#C4E6C1", "#6BA66B"),   # vert pastel      - traitements
    "model":   ("#D9CAEB", "#9A7CC2"),   # violet pastel    - modeles
    "metric":  ("#B4E3DB", "#57A296"),   # turquoise pastel - metriques / visualisation
    "result":  ("#F6CAD7", "#CC7C95"),   # rose pastel      - resultats / rapport
    "file":    ("#E5E5E8", "#9A9AA1"),   # gris clair       - fichiers / dossiers
    "manual":  ("#F7E4A9", "#C9A94E"),   # jaune pastel     - actions manuelles
}
BAND_FILL = "#F4F7FB"
BAND_EDGE = "#CCD8E4"
STORE_FILL = "#FAFAFB"
STORE_EDGE = "#C9C9CF"
BG = "#FFFFFF"
INK = "#2B2F36"        # couleur du texte principal
SUBINK = "#4A4F57"     # couleur du texte secondaire
EXEC_ARROW = "#5A6470"  # fleche pleine (ordre d'execution)
DATA_ARROW = "#8C95A2"  # fleche pointillee (flux de donnees)

BOXES = {}  # id -> (cx, cy, w, h)


def box(ax, bid, cx, cy, w, h, title, sub, kind, title_size=8.4, sub_size=7.0):
    """Dessine un bloc arrondi pastel (avec ombre douce) et l'enregistre."""
    fill, edge = COLORS[kind]
    # ombre portee discrete
    ax.add_patch(FancyBboxPatch((cx - w / 2 + 0.7, cy - h / 2 - 0.9), w, h,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                fc="#000000", ec="none", alpha=0.07, zorder=1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                fc=fill, ec=edge, lw=1.3, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.20, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=3)
        ax.text(cx, cy - h * 0.20, sub, ha="center", va="center",
                fontsize=sub_size, color=SUBINK, linespacing=1.25, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    BOXES[bid] = (cx, cy, w, h)


def anchor(bid, side):
    cx, cy, w, h = BOXES[bid]
    return {
        "T": (cx, cy + h / 2), "B": (cx, cy - h / 2),
        "L": (cx - w / 2, cy), "R": (cx + w / 2, cy),
    }[side]


def arrow(ax, a, sa, b, sb, kind="exec"):
    """Fleche pleine (execution) ou pointillee (flux de donnees) entre deux blocs."""
    p1, p2 = anchor(a, sa), anchor(b, sb)
    if kind == "exec":
        style = dict(color=EXEC_ARROW, lw=1.7, linestyle="-")
    else:
        style = dict(color=DATA_ARROW, lw=1.5, linestyle=(0, (4, 3)))
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 shrinkA=2, shrinkB=2, zorder=1.5,
                                 connectionstyle="arc3,rad=0", **style))


def phase_band(ax, y0, y1, label):
    ax.add_patch(FancyBboxPatch((5, y0), 146, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=2.5",
                                fc=BAND_FILL, ec=BAND_EDGE, lw=1.1, zorder=0))
    ax.text(9, y1 - 3.2, label, ha="left", va="center",
            fontsize=8.6, fontweight="bold", color="#3A4456", zorder=0.5)


def build():
    fig = plt.figure(figsize=(10.0, 15.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    # ----- Titre et sous-titre -----
    ax.text(100, 291.5, "Projet TAA - Schéma d'exécution des scripts",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(100, 283.5, "Pipeline Python - données, modèles, métriques et rapport",
            ha="center", va="center", fontsize=10.5, color="#6A7280")
    ax.plot([18, 182], [279, 279], color="#D5DCE5", lw=1.1, zorder=0)

    # ----- Colonne de stockage (a droite) -----
    ax.add_patch(FancyBboxPatch((155, 93), 42, 157,
                                boxstyle="round,pad=0,rounding_size=2.5",
                                fc=STORE_FILL, ec=STORE_EDGE, lw=1.1,
                                linestyle=(0, (5, 3)), zorder=0))
    ax.text(176, 245, "Stockage / fichiers", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color="#5A5F68")

    # ----- Lancement (action manuelle) + orchestrateur -----
    box(ax, "launch", 78, 272, 92, 9, "Utilisateur  -  python3 main.py", "",
        "manual", title_size=9)
    box(ax, "main", 78, 258, 78, 11.5, "main.py",
        "orchestrateur du pipeline - run_pipeline()", "script", title_size=9.2)
    arrow(ax, "launch", "B", "main", "T", "exec")

    # ====================================================================
    # Phase 0 - Preparation des donnees
    # ====================================================================
    phase_band(ax, 220, 246, "Phase 0 - Préparation des données")
    box(ax, "csv", 31, 231, 47, 15, "data/raw/", "credit_risk_dataset.csv", "file")
    box(ax, "loader", 84, 231, 48, 15, "data_loader.py", "load_raw_data()", "script")
    box(ax, "clean", 132, 231, 42, 15, "preprocessing.py", "clean_outliers()", "process")
    arrow(ax, "main", "B", "loader", "T", "exec")
    arrow(ax, "csv", "R", "loader", "L", "data")
    arrow(ax, "loader", "R", "clean", "L", "exec")

    # ====================================================================
    # Phase 1 - Pretraitement et decoupage
    # ====================================================================
    phase_band(ax, 184, 214, "Phase 1 - Prétraitement et découpage")
    box(ax, "encode", 132, 199, 42, 16, "preprocessing.py",
        "encode_categoricals()", "process")
    box(ax, "split", 84, 199, 48, 16, "split_data.py",
        "stratified_split()\n70 / 15 / 15", "process")
    box(ax, "scale", 31, 199, 47, 16, "preprocessing.py",
        "MedianImputer +\nStandardScaler (fit train)", "process")
    box(ax, "splits", 176, 199, 40, 18, "data/splits/",
        "train.csv\nval.csv\ntest.csv", "file", sub_size=6.6)
    arrow(ax, "clean", "B", "encode", "T", "exec")
    arrow(ax, "encode", "L", "split", "R", "exec")
    arrow(ax, "split", "L", "scale", "R", "exec")
    arrow(ax, "split", "R", "splits", "L", "data")

    # ====================================================================
    # Phase 2 - Entrainement des modeles
    # ====================================================================
    phase_band(ax, 150, 180, "Phase 2 - Entraînement des modèles (from scratch)")
    box(ax, "lr", 31, 165, 46, 16, "logistic_regression.py",
        "régression logistique\ndescente de gradient", "model", title_size=7.9)
    box(ax, "dt", 84, 165, 46, 16, "decision_tree.py",
        "arbre CART (Gini)\nprofondeur sur validation", "model", title_size=7.9)
    box(ax, "knn", 137, 165, 46, 16, "knn.py",
        "k plus proches voisins\nk choisi sur validation", "model", title_size=7.9)
    arrow(ax, "scale", "B", "lr", "T", "exec")
    arrow(ax, "scale", "B", "dt", "T", "exec")
    arrow(ax, "scale", "B", "knn", "T", "exec")

    # ====================================================================
    # Phase 3 - Evaluation et resultats
    # ====================================================================
    phase_band(ax, 96, 146, "Phase 3 - Évaluation, métriques et figures")
    box(ax, "metrics", 78, 128, 66, 19, "metrics.py",
        "accuracy, précision, rappel,\nF1-score, AUC-ROC,\nmatrice de confusion", "metric")
    box(ax, "viz", 78, 105, 66, 13, "visualization.py",
        "génération des figures (matplotlib)", "metric", sub_size=6.8)
    box(ax, "results", 176, 130, 40, 26, "results/",
        "metrics.json\npredictions.csv\nmodels/*.pkl\nsklearn_comparison.json *",
        "file", sub_size=6.3)
    ax.text(176, 115.5, "* option --compare-sklearn", ha="center", va="center",
            fontsize=5.6, style="italic", color="#8A8F98", zorder=3)
    box(ax, "figures", 176, 104, 40, 16, "reports/figures/",
        "7 figures .png\n(ROC, matrices,\ncomparaison...)", "file", sub_size=6.3)
    arrow(ax, "lr", "B", "metrics", "T", "exec")
    arrow(ax, "dt", "B", "metrics", "T", "exec")
    arrow(ax, "knn", "B", "metrics", "T", "exec")
    arrow(ax, "metrics", "B", "viz", "T", "exec")
    arrow(ax, "metrics", "R", "results", "L", "data")
    arrow(ax, "viz", "R", "figures", "L", "data")

    # ====================================================================
    # Phase 4 - Integration au rapport
    # ====================================================================
    phase_band(ax, 50, 90, "Phase 4 - Intégration au rapport LaTeX")
    box(ax, "tex", 120, 73, 56, 15, "reports/latex/",
        "main.tex + sections/", "result")
    box(ax, "pdflatex", 66, 73, 36, 15, "pdflatex", "compilation (x2)", "manual")
    box(ax, "pdf", 26, 73, 38, 15, "docs/rapport/", "rapport .pdf", "result")
    arrow(ax, "figures", "B", "tex", "T", "data")
    arrow(ax, "results", "B", "tex", "T", "data")
    arrow(ax, "tex", "L", "pdflatex", "R", "exec")
    arrow(ax, "pdflatex", "L", "pdf", "R", "data")

    # ====================================================================
    # Legende
    # ====================================================================
    ax.add_patch(FancyBboxPatch((5, 6), 192, 36,
                                boxstyle="round,pad=0,rounding_size=2.5",
                                fc="#FBFCFD", ec="#D5DCE5", lw=1.1, zorder=0))
    ax.text(9, 38, "Légende", ha="left", va="center", fontsize=9,
            fontweight="bold", color="#3A4456")

    # fleches
    ax.add_patch(FancyArrowPatch((12, 31), (30, 31), arrowstyle="-|>",
                                 mutation_scale=12, color=EXEC_ARROW, lw=1.7))
    ax.text(33, 31, "flèche pleine = ordre d'exécution", ha="left", va="center",
            fontsize=7.7, color=SUBINK)
    ax.add_patch(FancyArrowPatch((110, 31), (128, 31), arrowstyle="-|>",
                                 mutation_scale=12, color=DATA_ARROW, lw=1.5,
                                 linestyle=(0, (4, 3))))
    ax.text(131, 31, "flèche pointillée = flux de données / fichiers générés",
            ha="left", va="center", fontsize=7.7, color=SUBINK)

    # pastilles de couleur
    legend_items = [
        ("script", "script Python"),
        ("process", "traitement"),
        ("model", "modèle"),
        ("metric", "métriques / visualisation"),
        ("result", "résultats / rapport"),
        ("file", "fichiers / dossiers"),
        ("manual", "action manuelle"),
    ]
    x = 12
    y = 21
    col_w = 27
    per_row = 4
    for i, (kind, label) in enumerate(legend_items):
        col = i % per_row
        row = i // per_row
        cxp = x + col * (col_w + 20)
        cyp = y - row * 9
        fill, edge = COLORS[kind]
        ax.add_patch(FancyBboxPatch((cxp, cyp - 2.2), 7, 4.4,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=1.1))
        ax.text(cxp + 9, cyp, label, ha="left", va="center",
                fontsize=7.5, color=SUBINK)

    # ----- export -----
    base = OUT_DIR / "schema_execution_taa"
    fig.savefig(base.with_suffix(".svg"), format="svg", facecolor=BG)
    fig.savefig(base.with_suffix(".png"), format="png", dpi=200, facecolor=BG)
    fig.savefig(base.with_suffix(".pdf"), format="pdf", facecolor=BG)
    plt.close(fig)
    for ext in ("svg", "png", "pdf"):
        print("genere :", base.with_suffix("." + ext))


if __name__ == "__main__":
    build()
