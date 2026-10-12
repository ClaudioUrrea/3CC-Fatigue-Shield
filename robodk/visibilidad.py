# -*- coding: utf-8 -*-
"""
visibilidad.py - Ejes, targets y vistas de cámara de la escena HRC (sin reconstruirla).

Uso:
  python visibilidad.py mixto                 -> solo ejes de las bases de los robots
  python visibilidad.py paper                 -> sin ejes ni targets
  python visibilidad.py trabajo               -> todos los ejes y targets
  python visibilidad.py guardar_vista general -> guarda la cámara actual como "general"
  python visibilidad.py vista general         -> restaura la cámara "general"
  python visibilidad.py vistas                -> lista las vistas guardadas

Las vistas se guardan en vistas_camara.json: todas las figuras del paper se
pueden regenerar exactamente con el mismo encuadre.
"""
import json
import os
import sys
from robodk.robolink import (Robolink, ITEM_TYPE_FRAME, ITEM_TYPE_TARGET,
                             ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_OBJECT)
from robodk.robomath import Mat

EJES_MIXTO = ["HRC_H2017_Base", "HRC_A0509_Base"]
ARCHIVO_VISTAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vistas_camara.json")


def cargar_vistas():
    if os.path.isfile(ARCHIVO_VISTAS):
        with open(ARCHIVO_VISTAS, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def aplicar_modo(RDK, modo):
    todo = (modo == "trabajo")
    RDK.Render(False)
    for tipo in (ITEM_TYPE_FRAME, ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_OBJECT):
        for it in RDK.ItemList(tipo):
            ejes = todo or (modo == "mixto" and it.Name() in EJES_MIXTO)
            it.setVisible(True, ejes)
    for t in RDK.ItemList(ITEM_TYPE_TARGET):
        t.setVisible(todo)
    RDK.Render(True)
    print(f"Modo '{modo}' aplicado.")


args = sys.argv[1:] or ["mixto"]
accion = args[0]
RDK = Robolink()

if accion in ("mixto", "paper", "trabajo"):
    aplicar_modo(RDK, accion)
elif accion == "guardar_vista" and len(args) > 1:
    vistas = cargar_vistas()
    vistas[args[1]] = [list(fila) for fila in RDK.ViewPose().rows]   # matriz 4x4
    with open(ARCHIVO_VISTAS, "w", encoding="utf-8") as fh:
        json.dump(vistas, fh, indent=2)
    print(f"Vista '{args[1]}' guardada en {ARCHIVO_VISTAS}")
elif accion == "vista" and len(args) > 1:
    vistas = cargar_vistas()
    if args[1] not in vistas:
        sys.exit(f"No existe la vista '{args[1]}'. Disponibles: {list(vistas)}")
    RDK.setViewPose(Mat(vistas[args[1]]))
    RDK.Render(True)
    print(f"Vista '{args[1]}' restaurada.")
elif accion == "vistas":
    print("Vistas guardadas:", list(cargar_vistas()) or "ninguna")
else:
    sys.exit(__doc__)
