# -*- coding: utf-8 -*-
"""Estilo común de figuras: Palatino Linotype (Windows) o su clon libre
TeX Gyre Pagella / P052 / URW Palladio, también en las fórmulas."""
import matplotlib
from matplotlib import font_manager

CANDIDATAS = ["Palatino Linotype", "Palatino", "TeX Gyre Pagella", "P052",
              "URW Palladio L", "Book Antiqua"]


def aplicar():
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    fuente = next((f for f in CANDIDATAS if f in disponibles), None)
    if fuente is None:
        print("[AVISO] No se encontró Palatino ni un clon; se usa la serif por defecto.")
        fuente = "serif"
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [fuente] + CANDIDATAS,
        "mathtext.fontset": "custom",
        "mathtext.rm": fuente,
        "mathtext.it": f"{fuente}:italic",
        "mathtext.bf": f"{fuente}:bold",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return fuente
