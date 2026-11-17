# -*- coding: utf-8 -*-
"""Estilo común de las figuras del manuscrito.

Tipografía: Palatino Linotype (Windows) o su clon libre TeX Gyre Pagella /
P052 / URW Palladio, que es la misma familia Palatino que emplea la plantilla
MDPI, también dentro de las fórmulas.

Signos menos: se fuerza el uso de U+2212 (MINUS SIGN) en las etiquetas de los
ejes y en las expresiones matemáticas, en lugar del guion U+002D, que es más
corto y se alinea peor con las cifras.  La familia Palatino incluye el glifo;
`verificar_minus()` lo comprueba y avisa si la fuente elegida no lo tuviera.
"""
import unicodedata

import matplotlib
from matplotlib import font_manager

CANDIDATAS = ["Palatino Linotype", "Palatino", "TeX Gyre Pagella", "P052",
              "URW Palladio L", "Book Antiqua"]

MINUS = "−"          # MINUS SIGN, el que deben mostrar las figuras
HYPHEN = "-"         # HYPHEN-MINUS, el que se quiere evitar


def verificar_minus(fuente):
    """Avisa si la fuente elegida no contiene el glifo U+2212."""
    try:
        from fontTools.ttLib import TTFont
        ruta = font_manager.findfont(font_manager.FontProperties(family=fuente))
        if ord(MINUS) not in TTFont(ruta).getBestCmap():
            print(f"[AVISO] {fuente} no incluye U+2212; los signos menos "
                  f"podrían sustituirse por otra fuente.")
            return False
    except Exception:
        pass        # sin fontTools no se puede comprobar; no es un error
    return True


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
        "mathtext.default": "it",
        # U+2212 en lugar de U+002D en las etiquetas numéricas de los ejes
        "axes.unicode_minus": True,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    verificar_minus(fuente)
    return fuente


def auditar_pdf(ruta):
    """Informa de los signos menos y guiones que contiene un PDF ya generado.

    Devuelve (n_minus, n_hyphen).  Los guiones legítimos de palabras compuestas
    como ``3CC-r`` cuentan como U+002D y no son un problema; lo que no debe
    aparecer es un U+002D delante de una cifra.
    """
    import re
    import subprocess
    try:
        txt = subprocess.run(["pdftotext", ruta, "-"], capture_output=True,
                             text=True, check=True).stdout
    except Exception as exc:
        print(f"[AVISO] no se pudo auditar {ruta}: {exc}")
        return (0, 0)
    sospechosos = re.findall(r"-\s*\d", txt)
    print(f"{ruta}: U+2212={txt.count(MINUS)}  U+002D={txt.count(HYPHEN)}"
          + (f"  <-- {len(sospechosos)} guion(es) ante cifra" if sospechosos else ""))
    return (txt.count(MINUS), txt.count(HYPHEN))


if __name__ == "__main__":
    f = aplicar()
    print("Fuente:", f)
    print("U+2212:", unicodedata.name(MINUS))
