# -*- coding: utf-8 -*-
"""
crear_escena_hrc.py
-------------------
Construye en RoboDK una celda HRC multi-operador / multi-robot, parametrizada y
reproducible, para el artículo de Mathematics (MDPI):
  - 2 robots Doosan heterogéneos (H2017 carga pesada, A0509 precisión)
  - 3 operadores concurrentes con estado inicial distinto (fatiga, habilidad)
  - 5 estaciones (Interface, Sorting, Picking, Replenishing, Transport)
  - Transportador, zona de descanso, zonas de alcance y de separación (ISO/TS 15066)
  - Targets por estación, verificación de alcanzabilidad (IK) y de colisiones
  - Exporta todos los parámetros a JSON para el modelo matemático

Uso:
  1) Abrir RoboDK y crear una estación nueva vacía (File > New Station).
  2) pip install robodk
  3) Descargar de la biblioteca en línea de RoboDK los ficheros
     Doosan-Robotics-H2017.robot y Doosan-Robotics-A0509-White.robot y
     colocarlos en ./robots/ (o apuntar a su carpeta con la variable de
     entorno ROBODK_LIBRARY).  No se redistribuyen con este repositorio.
  4) python crear_escena_hrc.py
Todas las rutas son relativas a la carpeta de este script; la estación y el
JSON de parámetros se escriben en ../results/ (o en HRC_OUTPUT_DIR).
El script es idempotente: al re-ejecutarlo borra y reconstruye todo lo que
empiece con el prefijo "HRC_".
"""
import json
import math
import os
import sys

from robodk.robolink import (Robolink, ITEM_TYPE_ROBOT, COLLISION_ON)
from robodk.robomath import Mat, transl, rotx, rotz, invH, pi

# =============================================================================
# 1. PARÁMETROS (todo en mm y grados; origen = base del H2017)
# =============================================================================
PREFIJO = "HRC_"

# ---------------------------------------------------------------------------
# Rutas.  Todas son relativas a la carpeta de este script, de modo que el
# repositorio se pueda clonar y ejecutar en cualquier equipo sin editar nada.
# La salida se escribe en ../results/.  Los ficheros .robot de Doosan no se
# redistribuyen aquí (licencia de RoboDK): descárguelos de la biblioteca en
# línea de RoboDK y colóquelos en ./robots/, o indique otra carpeta con la
# variable de entorno ROBODK_LIBRARY.
# ---------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_SALIDA = os.environ.get("HRC_OUTPUT_DIR", os.path.join(AQUI, os.pardir, "results"))
DIR_ROBOTS = os.environ.get("ROBODK_LIBRARY", os.path.join(AQUI, "robots"))

RUTA_ESTACION = os.path.join(DIR_SALIDA, "Paper2_Mathematics.rdk")
RUTA_JSON = os.path.join(DIR_SALIDA, "escena_hrc_parametros.json")

ROBOTS = {
    "H2017": {"archivo": os.path.join(DIR_ROBOTS, "Doosan-Robotics-H2017.robot"),
              "base": (0, 0, 0), "rotz": 0, "gripper": "vacio",
              "alcance": 1700, "carga_kg": 20, "rol": "carga pesada"},
    "A0509": {"archivo": os.path.join(DIR_ROBOTS, "Doosan-Robotics-A0509-White.robot"),
              "base": (2600, 0, 950), "rotz": 0, "gripper": "pinza",
              "alcance": 900, "carga_kg": 5, "rol": "precision"},
}

# Estaciones: centro (x, y), tamaño (dx, dy), altura de mesa, robot que la atiende
ESTACIONES = {
    "Interface":    {"xy": (0, 1100),     "dim": (900, 600), "h": 750, "robot": "H2017", "tipo": "colaborativa"},
    "Replenishing": {"xy": (900, -700),   "dim": (700, 600), "h": 750, "robot": "H2017", "tipo": "robot"},
    "Transport":    {"xy": (-1000, -800), "dim": (1200, 800), "h": 400, "robot": "H2017", "tipo": "robot", "forma": "palet"},
    "Picking":      {"xy": (3100, 350),   "dim": (500, 400), "h": 750, "robot": "A0509", "tipo": "humana"},
    "Sorting":      {"xy": (3100, -350),  "dim": (500, 400), "h": 750, "robot": "A0509", "tipo": "humana"},
}

# Operadores: posición, orientación (grados), habilidad y fatiga inicial [0, 1].
# Los parámetros F (fatiga) y R (recuperación) del modelo de tres compartimentos
# (Xia & Frey-Law, 2008; Frey-Law et al., 2012) se dejan como marcadores:
# tómelos de la tabla publicada para la articulación elegida y verifíquelos.
# Perfil visual y antropométrico de cada operador:
#   estatura (mm), ancho (factor de hombros/cadera), piel, chaleco, casco (código
#   por habilidad: azul=novato, amarillo=intermedio, blanco=experto),
#   postura ("trabajo" o "cansado") y cabello largo (coleta) opcional.
PERFILES = {
    "Op1": {"estatura": 1820, "ancho": 1.05, "piel": [0.96, 0.80, 0.66, 1],
            "chaleco": [1.00, 0.90, 0.15, 1], "casco": [0.10, 0.35, 0.85, 1],
            "postura": "trabajo", "coleta": False},
    "Op2": {"estatura": 1640, "ancho": 0.90, "piel": [0.78, 0.57, 0.42, 1],
            "chaleco": [0.50, 0.28, 0.66, 1], "casco": [1.00, 0.85, 0.10, 1],
            "postura": "trabajo", "coleta": True},
    "Op3": {"estatura": 1740, "ancho": 1.00, "piel": [0.55, 0.38, 0.26, 1],
            "chaleco": [0.20, 0.45, 0.85, 1], "casco": [0.97, 0.97, 0.97, 1],
            "postura": "cansado", "coleta": False},
}

OPERADORES = {
    "Op1": {"xy": (0, 1750),      "rotz": -90, "habilidad": "novato",     "fatiga0": 0.15, "F": None, "R": None},
    "Op2": {"xy": (3650, -350),   "rotz": 180, "habilidad": "intermedio", "fatiga0": 0.32, "F": None, "R": None},
    "Op3": {"xy": (-1000, -1650), "rotz": 90,  "habilidad": "experto",    "fatiga0": 0.45, "F": None, "R": None},
}

# Opcional: ruta a un modelo humano externo (.stl/.obj/.sld) de pie, en mm, mirando a +X.
# Si es None, se genera un avatar articulado por código.
MODELO_HUMANO = None
ESCALA_MODELO = 1.0

ZONA_DESCANSO = {"xy": (-2800, 1800), "dim": (1400, 1200)}
TRANSPORTADOR = {"x": (800, 2250), "y": (-200, 200), "h": 750, "v_mm_s": 100,
                 "cajas_x": (1450, 1800, 2100)}   # material en tránsito sobre la banda

# Flujo de materiales: entrada en palet (Transport, H2017) -> transportador ->
# A0509 (Picking/Sorting con Op2) -> palet de salida. Reposición desde la estantería.
ENTORNO = {
    "estanteria": (800, 2000, -2200, -1800),            # x0, x1, y0, y1
    "palet_salida": (2000, 1250),                        # centro del europalet
    "armarios": {"H2017": (-2200, -200, 600, 500, 1100, "+x"),
                 "A0509": (2600, -1100, 450, 350, 700, "+y")},
    "pasillos_y": (2700, -2350),                         # líneas amarillas de pasillo
}
BANCO_A0509 = {"xy": (2600, 0), "dim": (300, 300), "h": 950}  # pedestal compacto, 200 mm sobre las mesas
PISO = {"x": (-6000, 7000), "y": (-4500, 4500)}
DIST_SEPARACION = 500   # distancia mínima de separación protectora (mm), ajustable
ALTURA_APROX = 180      # altura de aproximación sobre la mesa (mm)
ALTURA_TOMA = 100       # altura de toma sobre la mesa (mm)

# Umbrales del semáforo (idénticos al paper de 2025 para mantener consistencia)
UMBRAL_VERDE, UMBRAL_ROJO = 0.25, 0.40

COLOR = {
    "piso": [0.47, 0.50, 0.53, 1], "mesa": [0.45, 0.33, 0.22, 1],
    "banco": [0.2, 0.2, 0.2, 1], "transp": [0.15, 0.15, 0.15, 1],
    "descanso": [0.33, 0.40, 0.48, 1], "piel": [0.93, 0.76, 0.62, 1],
    "chaleco": [1.0, 0.45, 0.05, 1], "reflectante": [0.85, 0.85, 0.85, 1],
    "casco": [0.97, 0.97, 0.97, 1], "guante": [0.25, 0.25, 0.28, 1],
    "bota": [0.12, 0.10, 0.08, 1], "cinturon": [0.15, 0.15, 0.15, 1],
    "pantalon": [0.16, 0.22, 0.40, 1], "camisa": [0.95, 0.85, 0.2, 1],
    "alcance": [0.35, 0.75, 1.0, 0.13], "separacion": [1.0, 0.85, 0.15, 0.22],
    "verde": [0.1, 0.8, 0.1, 1], "naranja": [1.0, 0.55, 0.0, 1], "rojo": [0.9, 0.1, 0.1, 1],
}

# =============================================================================
# 2. GEOMETRÍA AUXILIAR (mallas de triángulos)
# =============================================================================
def malla_caja(x0, x1, y0, y1, z0, z1):
    """12 triángulos de un paralelepípedo alineado con los ejes."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    caras = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return [list(v[i]) for c in caras for i in c]


def malla_cilindro(cx, cy, r, z0, z1, n=48):
    pts = []
    for k in range(n):
        a0, a1 = 2 * pi * k / n, 2 * pi * (k + 1) / n
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
        p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        pts += [[cx, cy, z1], [p0[0], p0[1], z1], [p1[0], p1[1], z1]]          # tapa
        pts += [[cx, cy, z0], [p1[0], p1[1], z0], [p0[0], p0[1], z0]]          # fondo
        pts += [[p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z1]]    # lado
        pts += [[p0[0], p0[1], z0], [p1[0], p1[1], z1], [p0[0], p0[1], z1]]
    return pts


def malla_elipsoide(c, rad, nlat=14, nlon=24, lat_min=-90.0, lat_max=90.0):
    """Elipsoide (o casquete si se limita la latitud) centrado en c con radios rad."""
    pts = []
    def v(la, lo):
        la, lo = math.radians(la), math.radians(lo)
        return [c[0] + rad[0] * math.cos(la) * math.cos(lo),
                c[1] + rad[1] * math.cos(la) * math.sin(lo),
                c[2] + rad[2] * math.sin(la)]
    for i in range(nlat):
        a0 = lat_min + (lat_max - lat_min) * i / nlat
        a1 = lat_min + (lat_max - lat_min) * (i + 1) / nlat
        for k in range(nlon):
            o0, o1 = 360.0 * k / nlon, 360.0 * (k + 1) / nlon
            pts += [v(a0, o0), v(a0, o1), v(a1, o1), v(a0, o0), v(a1, o1), v(a1, o0)]
    return pts


def malla_segmento(p0, p1, r0, r1=None, n=18):
    """Tronco de cono entre p0 y p1 (extremidades), con extremos esféricos."""
    r1 = r0 if r1 is None else r1
    ax = [p1[i] - p0[i] for i in range(3)]
    L = math.sqrt(sum(x * x for x in ax)) or 1.0
    w = [x / L for x in ax]
    t = [1, 0, 0] if abs(w[0]) < 0.9 else [0, 1, 0]
    u = [w[1] * t[2] - w[2] * t[1], w[2] * t[0] - w[0] * t[2], w[0] * t[1] - w[1] * t[0]]
    nu = math.sqrt(sum(x * x for x in u))
    u = [x / nu for x in u]
    vv = [w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]]

    def anillo(p, r, a):
        return [p[i] + r * (math.cos(a) * u[i] + math.sin(a) * vv[i]) for i in range(3)]
    pts = []
    for k in range(n):
        a0, a1 = 2 * pi * k / n, 2 * pi * (k + 1) / n
        q0, q1 = anillo(p0, r0, a0), anillo(p0, r0, a1)
        s0, s1 = anillo(p1, r1, a0), anillo(p1, r1, a1)
        pts += [q0, q1, s1, q0, s1, s0]
    pts += malla_elipsoide(p0, (r0, r0, r0), 6, n)
    pts += malla_elipsoide(p1, (r1, r1, r1), 6, n)
    return pts


def malla_humano(perfil=None):
    """Avatar de operario industrial, frente hacia +X local, construido para 1750 mm
    y luego escalado según el perfil. Devuelve [(puntos, color), ...]."""
    C = dict(COLOR)
    perfil = perfil or {}
    for k in ("piel", "chaleco", "casco"):
        if k in perfil:
            C[k] = perfil[k]
    cansado = perfil.get("postura") == "cansado"
    inc = 45 if cansado else 0          # inclinación hacia adelante del tronco (mm)
    partes = []
    for lado in (1, -1):  # piernas y botas
        y = 105 * lado
        partes.append((malla_segmento([0, y, 930], [10, y, 500], 72, 60), C["pantalon"]))
        partes.append((malla_segmento([10, y, 500], [0, y, 110], 58, 48), C["pantalon"]))
        partes.append((malla_caja(-60, 170, y - 55, y + 55, 0, 110), C["bota"]))
    partes.append((malla_elipsoide([0, 0, 960], (125, 185, 110)), C["pantalon"]))
    partes.append((malla_elipsoide([0, 0, 1010], (132, 190, 28), 4, 24), C["cinturon"]))
    partes.append((malla_elipsoide([inc * 0.5, 0, 1230], (135, 205, 270)), C["chaleco"]))
    partes.append((malla_elipsoide([inc * 0.4, 0, 1180], (140, 210, 22), 4, 24), C["reflectante"]))
    partes.append((malla_elipsoide([inc * 0.6, 0, 1290], (140, 210, 22), 4, 24), C["reflectante"]))
    cab = [5 + inc * 2, 0, 1625 - inc]
    partes.append((malla_segmento([inc, 0, 1470], [cab[0] - 5, 0, cab[2] - 85], 48), C["piel"]))
    partes.append((malla_elipsoide(cab, (95, 88, 110)), C["piel"]))
    partes.append((malla_elipsoide([cab[0] - 5, 0, cab[2] + 35], (112, 108, 95), 8, 28, 0, 90), C["casco"]))
    partes.append((malla_elipsoide([cab[0] + 30, 0, cab[2] + 35], (150, 125, 12), 3, 28), C["casco"]))
    if perfil.get("coleta"):
        partes.append((malla_segmento([cab[0] - 95, 0, cab[2] + 10], [cab[0] - 120, 0, cab[2] - 170], 38, 22),
                       [0.18, 0.11, 0.07, 1]))
    for lado in (1, -1):  # brazos
        hombro = [inc, 215 * lado, 1430 - inc * 0.5]
        if cansado:   # brazos caídos, manos apoyadas en el borde de la mesa
            codo, mano = [inc + 60, 240 * lado, 1160], [inc + 230, 210 * lado, 960]
        else:
            codo, mano = [140, 235 * lado, 1180], [390, 170 * lado, 1080]
        partes.append((malla_elipsoide(hombro, (70, 70, 70)), C["chaleco"]))
        partes.append((malla_segmento(hombro, codo, 52, 45), C["chaleco"]))
        partes.append((malla_segmento(codo, mano, 42, 36), C["piel"]))
        partes.append((malla_elipsoide(mano, (55, 40, 32)), C["guante"]))
    # Escala antropométrica: altura y ancho de hombros/cadera
    sz = perfil.get("estatura", 1750) / 1750.0
    sy = perfil.get("ancho", 1.0)
    return [([[p[0] * sz, p[1] * sy * sz, p[2] * sz] for p in pts], col) for pts, col in partes]


def malla_toro(cz, R, r, n=40, m=12):
    """Anillo horizontal (toro) de radio mayor R y radio de tubo r a la altura cz."""
    def v(a, b):
        return [(R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a),
                cz + r * math.sin(b)]
    pts = []
    for i in range(n):
        a0, a1 = 2 * pi * i / n, 2 * pi * (i + 1) / n
        for k in range(m):
            b0, b1 = 2 * pi * k / m, 2 * pi * (k + 1) / m
            pts += [v(a0, b0), v(a1, b0), v(a1, b1), v(a0, b0), v(a1, b1), v(a0, b1)]
    return pts


def malla_gripper(tipo):
    """Geometría del efector en coordenadas de la brida (+Z hacia afuera).
    Devuelve (partes, z_tcp)."""
    oscuro, alu = [0.18, 0.18, 0.20, 1], [0.76, 0.78, 0.81, 1]
    if tipo == "vacio":      # ventosas para cajas (H2017)
        p = [(malla_segmento([0, 0, 0], [0, 0, 60], 55), alu),                   # adaptador
             (malla_caja(-160, 160, -110, 110, 60, 85), oscuro)]                 # placa
        copas = []
        for x in (-100, 100):
            for y in (-60, 60):
                copas += malla_segmento([x, y, 85], [x, y, 110], 12)             # vástago
                copas += malla_cilindro(x, y, 32, 110, 135, 24)                  # ventosa
        p.append((copas, [0.10, 0.10, 0.10, 1]))
        return p, 135
    # pinza paralela de dos dedos (A0509)
    p = [(malla_segmento([0, 0, 0], [0, 0, 25], 40), alu),
         (malla_caja(-45, 45, -35, 35, 25, 100), oscuro)]
    dedos = malla_caja(-12, 12, 40, 55, 100, 160) + malla_caja(-12, 12, -55, -40, 100, 160)
    dedos += malla_caja(-12, 12, 28, 55, 100, 115) + malla_caja(-12, 12, -55, -28, 100, 115)
    p.append((dedos, alu))
    return p, 150


def agregar_multicolor(RDK, nombre, partes, padre=None):
    """Crea UN objeto con varias formas, cada una con su color."""
    datos = []
    for pts, col in partes:
        datos += [Mat(pts).tr(), col]
    obj = RDK.AddShape(datos)
    obj.setName(PREFIJO + nombre)
    if padre is not None:
        obj.setParent(padre)
    return obj


def agregar_forma(RDK, nombre, puntos, color, padre=None):
    obj = RDK.AddShape(Mat(puntos).tr())
    obj.setName(PREFIJO + nombre)
    obj.setColor(color)
    if padre is not None:
        obj.setParent(padre)
    return obj


def mesa(RDK, nombre, cx, cy, dx, dy, h, color):
    return agregar_forma(RDK, nombre, malla_caja(cx - dx / 2, cx + dx / 2,
                                                 cy - dy / 2, cy + dy / 2, 0, h), color)


def color_semaforo(f):
    if f < UMBRAL_VERDE:
        return COLOR["verde"]
    return COLOR["naranja"] if f < UMBRAL_ROJO else COLOR["rojo"]


# =============================================================================
# 2b. ENTORNO INDUSTRIAL (mobiliario, flujo de materiales, piezas)
# =============================================================================
C_ENT = {
    "madera": [0.62, 0.50, 0.36, 1], "acero": [0.32, 0.32, 0.35, 1],
    "aluminio": [0.76, 0.78, 0.81, 1], "banda": [0.08, 0.08, 0.09, 1],
    "carton": [0.74, 0.57, 0.37, 1], "cinta": [0.86, 0.78, 0.60, 1],
    "palet": [0.80, 0.66, 0.45, 1], "klt": [0.12, 0.35, 0.75, 1],
    "klt_int": [0.07, 0.20, 0.45, 1], "gris_claro": [0.86, 0.86, 0.86, 1],
    "panel": [0.15, 0.17, 0.20, 1], "motor": [0.15, 0.30, 0.60, 1],
    "rack_poste": [0.10, 0.30, 0.70, 1], "rack_viga": [0.72, 0.74, 0.77, 1],
    "linea": [1.00, 0.85, 0.00, 1], "pieza_roja": [0.80, 0.15, 0.12, 1],
    "pieza_verde": [0.15, 0.60, 0.25, 1], "asiento": [0.20, 0.35, 0.55, 1],
}


def malla_mesa(cx, cy, dx, dy, h, esp=40, pata=50):
    """Mesa de trabajo: cubierta de madera y patas de acero."""
    x0, x1, y0, y1 = cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2
    partes = [(malla_caja(x0, x1, y0, y1, h - esp, h), C_ENT["madera"])]
    patas = []
    for px in (x0 + 30, x1 - 30 - pata):
        for py in (y0 + 30, y1 - 30 - pata):
            patas += malla_caja(px, px + pata, py, py + pata, 0, h - esp)
    patas += malla_caja(x0 + 30, x1 - 30, y0 + 30, y0 + 30 + pata, 120, 160)   # travesaños
    patas += malla_caja(x0 + 30, x1 - 30, y1 - 30 - pata, y1 - 30, 120, 160)
    partes.append((patas, C_ENT["acero"]))
    return partes


def malla_carton(x0, y0, z0, dx, dy, dz):
    return [(malla_caja(x0, x0 + dx, y0, y0 + dy, z0, z0 + dz), C_ENT["carton"]),
            (malla_caja(x0 + dx / 2 - 25, x0 + dx / 2 + 25, y0 - 1, y0 + dy + 1, z0 + dz, z0 + dz + 2),
             C_ENT["cinta"])]


def malla_klt(cx, cy, z0, dx, dy, dz, contenido=None):
    """Caja plástica tipo KLT abierta, opcionalmente con piezas visibles."""
    x0, x1, y0, y1 = cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2
    p = [(malla_caja(x0, x1, y0, y1, z0, z0 + dz), C_ENT["klt"]),
         (malla_caja(x0 + 12, x1 - 12, y0 + 12, y1 - 12, z0 + dz - 30, z0 + dz + 1), C_ENT["klt_int"])]
    if contenido:
        for i in range(3):
            for k in range(2):
                px = x0 + 25 + i * (dx - 50) / 2.5
                py = y0 + 25 + k * (dy - 50) / 1.5
                p.append((malla_caja(px, px + 28, py, py + 28, z0 + dz - 28, z0 + dz + 4), contenido))
    return p


def malla_palet(cx, cy, capas=1, col_extra=0):
    """Europalet 1200x800x144 con cajas de cartón (3x2 por capa)."""
    x0, y0 = cx - 600, cy - 400
    p = []
    madera = []
    for yy in (y0, cy - 50, y0 + 700):
        madera += malla_caja(x0, x0 + 1200, yy, yy + 100, 0, 22)                  # patines
        for xx in (x0, cx - 72, x0 + 1055):
            madera += malla_caja(xx, xx + 145, yy, yy + 100, 22, 100)             # tacos
    for k in range(7):
        xx = x0 + k * (1200 - 145) / 6
        madera += malla_caja(xx, xx + 145, y0, y0 + 800, 100, 144)                # tablas
    p.append((madera, C_ENT["palet"]))
    for capa in range(capas):
        for i in range(3):
            for j in range(2):
                p += malla_carton(x0 + i * 400 + 5, y0 + j * 400 + 5, 144 + capa * 256, 390, 390, 250)
    for j in range(col_extra):  # columna parcial adicional (palet a medio consumir)
        p += malla_carton(x0 + 5, y0 + j * 400 + 5, 144 + capas * 256, 390, 390, 250)
    return p


def malla_transportador(t):
    x0, x1 = t["x"]
    y0, y1 = t["y"]
    h = t["h"]
    p = [(malla_caja(x0, x1, y0 + 25, y1 - 25, h - 40, h), C_ENT["banda"])]
    rieles = (malla_caja(x0, x1, y1 - 25, y1, h - 90, h + 30) +
              malla_caja(x0, x1, y0, y0 + 25, h - 90, h + 30))
    p.append((rieles, C_ENT["aluminio"]))
    rod = (malla_segmento([x0, y0 + 25, h - 20], [x0, y1 - 25, h - 20], 22) +
           malla_segmento([x1, y0 + 25, h - 20], [x1, y1 - 25, h - 20], 22))
    p.append((rod, C_ENT["gris_claro"]))
    patas = []
    for px in (x0 + 40, (x0 + x1) / 2, x1 - 90):
        for py in (y0 + 10, y1 - 60):
            patas += malla_caja(px, px + 50, py, py + 50, 0, h - 90)
    patas += malla_caja(x0 + 40, x1 - 40, y0 + 10, y0 + 60, 150, 190)
    patas += malla_caja(x0 + 40, x1 - 40, y1 - 60, y1 - 10, 150, 190)
    p.append((patas, C_ENT["acero"]))
    p.append((malla_caja(x1 - 230, x1 - 70, y1, y1 + 120, h - 200, h - 60), C_ENT["motor"]))   # motorreductor
    p.append((malla_segmento([x1 - 150, y1 + 120, h - 130], [x1 - 150, y1 + 200, h - 130], 55),
              C_ENT["motor"]))                                                                  # motor
    for xc in t.get("cajas_x", []):   # material en tránsito
        p += malla_carton(xc - 125, -100, h, 250, 200, 150)
    return p


def malla_estanteria(x0, x1, y0, y1, niveles=(300, 900, 1500), alto=1800):
    p = []
    postes = []
    for px in (x0, x1 - 70):
        for py in (y0, y1 - 70):
            postes += malla_caja(px, px + 70, py, py + 70, 0, alto)
    p.append((postes, C_ENT["rack_poste"]))
    vigas, bandejas = [], []
    for z in niveles:
        vigas += malla_caja(x0, x1, y0, y0 + 70, z - 90, z) + malla_caja(x0, x1, y1 - 70, y1, z - 90, z)
        bandejas += malla_caja(x0 + 70, x1 - 70, y0 + 70, y1 - 70, z - 20, z)
    p.append((vigas, C_ENT["rack_viga"]))
    p.append((bandejas, C_ENT["gris_claro"]))
    ancho = (x1 - x0 - 200) / 3
    for n, z in enumerate(niveles):
        for i in range(3):
            xc = x0 + 100 + ancho * (i + 0.5)
            if (n + i) % 2 == 0:
                p += malla_klt(xc, (y0 + y1) / 2, z, ancho - 60, (y1 - y0) - 120, 220,
                               C_ENT["pieza_roja"] if i == 1 else C_ENT["gris_claro"])
            else:
                p += malla_carton(xc - (ancho - 60) / 2, y0 + 60, z, ancho - 60, (y1 - y0) - 120, 300)
    return p


def malla_armario(cx, cy, dx, dy, alto, frente="+x"):
    """Controlador del robot: armario gris con panel frontal."""
    x0, x1, y0, y1 = cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2
    p = [(malla_caja(x0, x1, y0, y1, 0, alto), C_ENT["gris_claro"])]
    if frente == "+x":
        p.append((malla_caja(x1, x1 + 4, y0 + 60, y1 - 60, alto * 0.45, alto - 80), C_ENT["panel"]))
    else:
        p.append((malla_caja(x0 + 60, x1 - 60, y1, y1 + 4, alto * 0.45, alto - 80), C_ENT["panel"]))
    return p


def malla_zona_descanso(cx, cy):
    p = [(malla_caja(cx - 550, cx + 450, cy + 250, cy + 600, 420, 460), C_ENT["asiento"])]
    patas = []
    for px in (cx - 520, cx + 380):
        patas += malla_caja(px, px + 40, cy + 280, cy + 570, 0, 420)
    p.append((patas, C_ENT["acero"]))
    p.append((malla_cilindro(cx, cy - 150, 380, 710, 740, 36), C_ENT["gris_claro"]))
    p.append((malla_segmento([cx, cy - 150, 0], [cx, cy - 150, 710], 35), C_ENT["acero"]))
    return p


def malla_piezas_estacion(nombre, cx, cy, h):
    """Material de trabajo sobre cada estación (la pieza central queda bajo el target)."""
    if nombre == "Interface":   # fijación de ensamblaje + carcasa a montar
        return [(malla_caja(cx - 200, cx + 200, cy - 150, cy + 150, h, h + 20), C_ENT["aluminio"]),
                (malla_cilindro(cx, cy, 60, h + 20, h + 85, 32), C_ENT["gris_claro"])] + \
               malla_klt(cx + 330, cy - 180, h, 180, 160, 100, C_ENT["pieza_verde"])
    if nombre == "Replenishing":  # bandeja de reposición + dos KLT
        return [(malla_caja(cx - 140, cx + 140, cy - 100, cy + 100, h, h + 50), C_ENT["klt"])] + \
               malla_klt(cx + 240, cy - 200, h, 200, 160, 120, C_ENT["pieza_roja"]) + \
               malla_klt(cx + 240, cy + 200, h, 200, 160, 120, C_ENT["gris_claro"])
    if nombre in ("Picking", "Sorting"):  # pieza a manipular + cajas de clasificación
        col = C_ENT["pieza_roja"] if nombre == "Picking" else C_ENT["pieza_verde"]
        return [(malla_caja(cx - 30, cx + 30, cy - 30, cy + 30, h, h + 60), col)] + \
               malla_klt(cx + 180, cy - 90, h, 120, 140, 100, col) + \
               malla_klt(cx + 180, cy + 90, h, 120, 140, 100, C_ENT["gris_claro"])
    return []


def crear_entorno(RDK):
    """Agrega todo el mobiliario y material; devuelve la lista de objetos creados."""
    objs = []
    t = TRANSPORTADOR
    objs.append(agregar_multicolor(RDK, "Transportador", malla_transportador(t)))
    e = ENTORNO
    objs.append(agregar_multicolor(RDK, "Estanteria", malla_estanteria(*e["estanteria"])))
    objs.append(agregar_multicolor(RDK, "Palet_Salida", malla_palet(*e["palet_salida"], capas=1, col_extra=2)))
    for nombre, (cx, cy, dx, dy, alto, fr) in e["armarios"].items():
        objs.append(agregar_multicolor(RDK, "Controlador_" + nombre, malla_armario(cx, cy, dx, dy, alto, fr)))
    lineas = []
    for y in e["pasillos_y"]:
        lineas += malla_caja(PISO["x"][0] + 100, PISO["x"][1] - 100, y - 40, y + 40, 1, 3)
    objs.append(agregar_forma(RDK, "LineasPasillo", lineas, C_ENT["linea"]))
    z = ZONA_DESCANSO
    objs.append(agregar_multicolor(RDK, "Mobiliario_Descanso", malla_zona_descanso(*z["xy"])))
    return objs


# =============================================================================
# 3. CONSTRUCCIÓN
# =============================================================================
def limpiar(RDK):
    """Borra solo los elementos HRC_ de nivel superior (sus hijos se borran con ellos)."""
    raiz = []
    for it in RDK.ItemList():
        if it.Name().startswith(PREFIJO) and not it.Parent().Name().startswith(PREFIJO):
            raiz.append(it)
    for it in raiz:
        it.Delete()


def diagnosticar(RDK, robot, q0, q1, pasos=25):
    """Recorre el tramo articular q0->q1 y devuelve los objetos en colisión."""
    en_choque = set()
    for k in range(pasos + 1):
        q = [a + (b - a) * k / pasos for a, b in zip(q0, q1)]
        robot.setJoints(q)
        if RDK.Collisions() > 0:
            en_choque.update(it.Name() for it in RDK.CollisionItems())
    return sorted(en_choque)


def cargar_robot(RDK, clave, cfg):
    nombre = PREFIJO + clave
    if os.path.isfile(cfg["archivo"]):
        robot = RDK.AddFile(cfg["archivo"])
    else:  # reutiliza uno ya presente en la estación
        candidatos = [r for r in RDK.ItemList(ITEM_TYPE_ROBOT) if clave in r.Name()]
        if not candidatos:
            sys.exit(f"[ERROR] No se encontró {cfg['archivo']} ni un robot '{clave}' en la estación.\n"
                     "Descárguelo desde la biblioteca en línea de RoboDK y corrija la ruta.")
        robot = candidatos[0]
    robot.setName(nombre)
    base = robot.Parent()
    base.setName(nombre + "_Base")
    x, y, z = cfg["base"]
    base.setPose(transl(x, y, z) * rotz(math.radians(cfg["rotz"])))
    robot.setJoints(robot.JointsHome())
    # Efector final con geometría real; el TCP coincide con la punta del efector
    partes, z_tcp = malla_gripper(cfg["gripper"])
    herramienta = robot.AddTool(transl(0, 0, z_tcp), nombre + "_Gripper")
    # La geometría se adjunta al TCP como objeto hijo (se mueve con el robot);
    # se define en coordenadas de brida, por eso se desplaza -z_tcp.
    cuerpo = agregar_multicolor(RDK, clave + "_GripperGeom", partes)
    cuerpo.setParent(herramienta)
    cuerpo.setPose(transl(0, 0, -z_tcp))
    cfg["tcp_z"] = z_tcp
    robot.setPoseTool(herramienta)
    robot.gripper_geom = cuerpo
    return robot, base, herramienta


# Andon (torre de señalización) de fatiga: poste anclado al piso junto a cada
# operador, con tres luces apiladas (rojo arriba, naranja al medio, verde abajo).
LUCES_ANDON = ("Verde", "Naranja", "Roja")
# Posición del andon relativa a cada operador (x = frente, y = izquierda), lateral
# y hacia afuera de la celda para no tapar al operador en las vistas típicas.
ANDON_POS_LOCAL = {"Op1": (-100, -600), "Op2": (-100, 600), "Op3": (-100, -600)}
ANDON_ALTURA = 1900               # altura del poste (mm)


def nivel_semaforo(f):
    if f < UMBRAL_VERDE:
        return "Verde"
    return "Naranja" if f < UMBRAL_ROJO else "Roja"


def color_luz(nivel, encendida):
    base = {"Verde": COLOR["verde"], "Naranja": COLOR["naranja"], "Roja": COLOR["rojo"]}[nivel]
    if encendida:
        return base
    return [0.25 * base[0] + 0.12, 0.25 * base[1] + 0.12, 0.25 * base[2] + 0.12, 1]


def crear_andon(RDK, clave, marco, fatiga0):
    x, y = ANDON_POS_LOCAL.get(clave, (-100, -600))
    h = ANDON_ALTURA
    oscuro, acero = [0.15, 0.15, 0.17, 1], [0.55, 0.56, 0.58, 1]
    partes = [(malla_cilindro(x, y, 130, 0, 15, 32), oscuro),                     # placa base
              (malla_segmento([x, y, 15], [x, y, h], 22), acero),                   # poste
              (malla_cilindro(x, y, 55, h, h + 30, 32), oscuro),                    # zócalo
              (malla_cilindro(x, y, 55, h + 3 * 95 + 30, h + 3 * 95 + 55, 32), oscuro)]  # tapa
    agregar_multicolor(RDK, clave + "_Andon", partes, marco)
    activa = nivel_semaforo(fatiga0)
    for i, nivel in enumerate(LUCES_ANDON):
        z0 = h + 30 + i * 95
        agregar_forma(RDK, f"{clave}_Luz_{nivel}", malla_cilindro(x, y, 50, z0 + 3, z0 + 92, 32),
                      color_luz(nivel, nivel == activa), marco)


def crear_operador(RDK, clave, cfg):
    x, y = cfg["xy"]
    marco = RDK.AddFrame(PREFIJO + clave)
    marco.setPose(transl(x, y, 0) * rotz(math.radians(cfg["rotz"])))
    if MODELO_HUMANO and os.path.isfile(MODELO_HUMANO):
        cuerpo = RDK.AddFile(MODELO_HUMANO, marco)
        cuerpo.setName(PREFIJO + clave + "_Cuerpo")
        if ESCALA_MODELO != 1.0:
            cuerpo.Scale(ESCALA_MODELO)
    else:
        agregar_multicolor(RDK, clave + "_Cuerpo", malla_humano(PERFILES.get(clave)), marco)
    crear_andon(RDK, clave, marco, cfg["fatiga0"])
    agregar_forma(RDK, clave + "_ZonaSeparacion",
                  malla_cilindro(0, 0, 300 + DIST_SEPARACION, 10, 14), COLOR["separacion"], marco)
    return marco


def normalizar_q(robot, q, ref):
    """Suma/resta 360° a cada eje para acercarlo a ref, respetando los límites."""
    lo, hi, _ = robot.JointLimits()
    lo, hi = lo.list(), hi.list()
    out = []
    for i, (a, r) in enumerate(zip(q, ref)):
        mejor = a
        for cand in (a - 360, a + 360):
            if lo[i] <= cand <= hi[i] and abs(cand - r) < abs(mejor - r):
                mejor = cand
        out.append(mejor)
    return out


def construir():
    RDK = Robolink()
    RDK.Render(False)
    limpiar(RDK)
    print("Construyendo la celda...")

    # --- Entorno fijo ---
    piso = agregar_forma(RDK, "Piso", malla_caja(*PISO["x"], *PISO["y"], -20, 0), COLOR["piso"])
    entorno = crear_entorno(RDK)
    b = BANCO_A0509
    banco = mesa(RDK, "Banco_A0509", *b["xy"], *b["dim"], b["h"], COLOR["banco"])
    z = ZONA_DESCANSO
    descanso = agregar_forma(RDK, "ZonaDescanso",
                             malla_caja(z["xy"][0] - z["dim"][0] / 2, z["xy"][0] + z["dim"][0] / 2,
                                        z["xy"][1] - z["dim"][1] / 2, z["xy"][1] + z["dim"][1] / 2, 0, 20),
                             COLOR["descanso"])

    # --- Robots y zonas de alcance ---
    robots, zonas = {}, []
    for clave, cfg in ROBOTS.items():
        robot, base, herr = cargar_robot(RDK, clave, cfg)
        robots[clave] = (robot, base, herr)
        x, y, zb = cfg["base"]
        zonas.append(agregar_forma(RDK, clave + "_ZonaAlcance",
                                   malla_cilindro(x, y, cfg["alcance"], 4, 8), COLOR["alcance"]))

    # --- Estaciones, mesas y targets ---
    info_targets = {}
    for nombre, e in ESTACIONES.items():
        cx, cy = e["xy"]
        if e.get("forma") == "palet":   # palet de entrada: la capa superior queda a h
            agregar_multicolor(RDK, "Palet_" + nombre, malla_palet(cx, cy, capas=1, col_extra=0))
        else:
            agregar_multicolor(RDK, "Mesa_" + nombre, malla_mesa(cx, cy, *e["dim"], e["h"]))
            agregar_multicolor(RDK, "Piezas_" + nombre, malla_piezas_estacion(nombre, cx, cy, e["h"]))
        marco = RDK.AddFrame(PREFIJO + "Est_" + nombre)
        marco.setPose(transl(cx, cy, 0))
        robot, base, herr = robots[e["robot"]]
        info_targets[nombre] = []
        for etiqueta, dz in (("Aprox", ALTURA_APROX), ("Toma", ALTURA_TOMA)):
            tgt = RDK.AddTarget(f"{PREFIJO}T_{nombre}_{etiqueta}", marco, robot)
            tgt.setPose(transl(0, 0, e["h"] + dz) * rotx(pi))
            ref = invH(base.PoseAbs()) * marco.PoseAbs()
            # Solución IK más cercana a Home y con ángulos llevados a +-180° de Home:
            # evita que la interpolación articular dé la vuelta larga (p. ej. J1 por 180°)
            home = robot.JointsHome().list()
            j = robot.SolveIK(tgt.Pose(), home, herr.PoseTool(), ref)
            ok = len(j.list()) >= 6
            if ok:
                j = Mat(normalizar_q(robot, j.list(), home))
            if ok:
                # Target articular: el programa usa exactamente la solución IK validada
                tgt.setJoints(j.list())
                tgt.setAsJointTarget()
            info_targets[nombre].append({"target": tgt.Name(), "alcanzable": ok,
                                         "q": j.list() if ok else None})
            print(f"  {tgt.Name():32s} -> {'OK' if ok else 'NO ALCANZABLE: ajuste xy/h'}")

    # --- Operadores ---
    for clave, cfg in OPERADORES.items():
        crear_operador(RDK, clave, cfg)

    # --- Colisiones: activar y excluir solo la interfaz de montaje ---
    # Los eslabones 0 (base fija) y 1 (base giratoria) apoyan sobre el piso o el
    # pedestal por diseño; ese contacto no es una colisión física real.
    RDK.setCollisionActive(COLLISION_ON)
    for clave, (robot, base, herr) in robots.items():
        for obj in [piso, banco, descanso] + zonas:
            for eslabon in (0, 1):
                RDK.setCollisionActivePair(0, robot, obj, eslabon, 0)
        # El gripper está montado en la brida: no es colisión con su propio robot
        for eslabon in range(7):
            RDK.setCollisionActivePair(0, robot, robot.gripper_geom, eslabon, 0)
    # Verificación en Home, una vez definidas TODAS las exclusiones
    for clave, (robot, base, herr) in robots.items():
        robot.setJoints(robot.JointsHome())
    if RDK.Collisions() > 0:
        print(f"  [AVISO] Colisión con los robots en Home: "
              f"{sorted(it.Name() for it in RDK.CollisionItems())}")
    else:
        print("  Robots en Home: sin colisiones")

    # --- Programas: Home -> (Aprox -> Toma -> Aprox) por estación -> Home ---
    colisiones, info_movimientos = {}, {}
    for clave, (robot, base, herr) in robots.items():
        prog = RDK.AddProgram(PREFIJO + "Prog_" + clave, robot)
        prog.setPoseTool(herr)
        home = robot.JointsHome().list()
        secuencia = [("Home", home)]
        for nombre, e in ESTACIONES.items():
            if e["robot"] != clave:
                continue
            aprox, toma = info_targets[nombre]
            if not (aprox["alcanzable"] and toma["alcanzable"]):
                print(f"  [AVISO] {nombre} se omite del programa: target no alcanzable")
                continue
            secuencia += [(aprox["target"], aprox["q"]), (toma["target"], toma["q"]),
                          (aprox["target"], aprox["q"])]
        secuencia.append(("Home", home))
        # Planificación de cada tramo con selección de movimiento libre de colisión:
        #   dentro de una estación (Aprox <-> Toma): MoveL vertical, si no MoveJ;
        #   entre estaciones: MoveJ, si no MoveL, si no desvío por Home.
        robot.setPoseFrame(base)
        robot.setPoseTool(herr)
        tramos, movimientos = [], []

        def pose_rel(nombre):
            return invH(base.PoseAbs()) * RDK.Item(nombre).PoseAbs()

        def prueba(tipo, qa, qb, nb):
            if tipo == "MoveJ":
                return robot.MoveJ_Test(qa, qb)
            return robot.MoveL_Test(qa, pose_rel(nb))

        for (n0, q0), (n1, q1) in zip(secuencia[:-1], secuencia[1:]):
            misma_est = n0 != "Home" and n1 != "Home" and n0.rsplit("_", 1)[0] == n1.rsplit("_", 1)[0]
            if "Home" in (n0, n1):
                opciones = ["MoveJ"]
            elif misma_est:
                opciones = ["MoveL", "MoveJ"]
            else:
                opciones = ["MoveJ", "MoveL"]
            elegido = None
            for tipo in opciones:
                if prueba(tipo, q0, q1, n1) == 0:
                    elegido = tipo
                    break
            destino = q1 if n1 == "Home" else RDK.Item(n1)
            if elegido == "MoveJ":
                prog.MoveJ(destino)
            elif elegido == "MoveL":
                prog.MoveL(destino)
            elif robot.MoveJ_Test(q0, home) == 0 and robot.MoveJ_Test(home, q1) == 0:
                elegido = "MoveJ via Home"
                prog.MoveJ(home)
                prog.MoveJ(destino)
            else:
                prog.MoveJ(destino)
                objs = diagnosticar(RDK, robot, q0, q1)
                tramos.append({"de": n0, "a": n1, "objetos": objs})
                print(f"  [COLISION] {clave}: {n0} -> {n1}  con {objs}")
            movimientos.append({"de": n0, "a": n1, "tipo": elegido or "sin solucion"})
            if elegido and elegido != opciones[0]:
                print(f"  [INFO] {clave}: {n0} -> {n1} resuelto con {elegido}")
        info_movimientos[clave] = movimientos
        robot.setJoints(home)
        colisiones[clave] = tramos
        print(f"  Programa {prog.Name()}: tramos con colisión = {len(tramos)}")

    RDK.Render(True)
    RDK.Save(RUTA_ESTACION)
    print(f"Estación guardada en {RUTA_ESTACION}")

    # --- Exportar parámetros para el modelo matemático ---
    datos = {"robots": ROBOTS, "estaciones": ESTACIONES, "operadores": OPERADORES,
             "perfiles": PERFILES,
             "zona_descanso": ZONA_DESCANSO, "transportador": TRANSPORTADOR, "entorno": ENTORNO,
             "dist_separacion_mm": DIST_SEPARACION,
             "umbrales_semaforo": [UMBRAL_VERDE, UMBRAL_ROJO],
             "targets": info_targets, "colisiones_prueba": colisiones,
             "movimientos": info_movimientos}
    os.makedirs(os.path.dirname(RUTA_JSON), exist_ok=True)
    with open(RUTA_JSON, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False)
    print(f"Parámetros exportados a {RUTA_JSON}")


# =============================================================================
# 4. UTILIDAD PARA LA SIMULACIÓN (importable desde otros scripts)
# =============================================================================
def actualizar_semaforo(RDK, operador, fatiga):
    """Enciende la luz del andon del operador según su fatiga en [0, 1]
    (verde < 0.25 <= naranja < 0.40 <= rojo) y atenúa las otras dos."""
    activa = nivel_semaforo(fatiga)
    for nivel in LUCES_ANDON:
        luz = RDK.Item(f"{PREFIJO}{operador}_Luz_{nivel}")
        if luz.Valid():
            luz.setColor(color_luz(nivel, nivel == activa))


if __name__ == "__main__":
    construir()
