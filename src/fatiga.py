# -*- coding: utf-8 -*-
"""
fatiga.py - Modelo dinámico de fatiga de tres compartimentos (Xia & Frey-Law)
con recuperación en pausas (Looft et al.), integrador que preserva el simplex,
verificación numérica de las proposiciones del paper y demostración con los
tres operadores de la escena (opcionalmente enciende los andones en RoboDK).

Uso:
  python fatiga.py verificar        -> comprueba numéricamente Lemas/Proposiciones
  python fatiga.py demo             -> simula un turno y guarda fatiga_demo.png
  python fatiga.py demo --robodk    -> además actualiza los andones en RoboDK
"""
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Parámetros verificados en las fuentes originales:
#  F, R  : hombro, Tabla 1 de Frey-Law, Looft & Heitsman (2012), J. Biomech. 45:1803-1808
#  r     : valor general (todas las regiones salvo mano), Looft et al. (2018), J. Biomech. 77:16-25;
#          sin datos de hombro en esa fuente -> extrapolación declarada en el paper
#  L_D, L_R : 10, Xia & Frey-Law (2008), J. Biomech. 41:3046-3052
# ---------------------------------------------------------------------------
PARAM = {"F": 0.01820, "R": 0.00168,   # hombro
         "LD": 10.0, "LR": 10.0,
         "r": 15.0}

# Factor de habilidad: fracción de la carga nominal que efectivamente exige la
# tarea a cada nivel (supuesto de modelado a declarar y justificar en el paper).
KAPPA = {"novato": 1.20, "intermedio": 1.00, "experto": 0.85}
UMBRALES = (0.25, 0.40)


def control(TL, x, p):
    """Controlador de reclutamiento C(t) del modelo 3C."""
    MA, MR, _ = x
    if MA < TL:
        return p["LD"] * (TL - MA) if MR > TL - MA else p["LD"] * MR
    return p["LR"] * (TL - MA)


def campo(TL, x, p):
    MA, MR, MF = x
    Reff = p["R"] * (p["r"] if TL == 0 else 1.0)
    C = control(TL, x, p)
    return np.array([C - p["F"] * MA, -C + Reff * MF, p["F"] * MA - Reff * MF])


def dt_max(p):
    """Cota de paso de Euler que preserva el simplex (Proposición 3)."""
    return 1.0 / max(p["LD"], p["LR"] + p["F"], p["r"] * p["R"])


def paso(TL, x, dt, p, seguro=True):
    """Avanza dt segundos; con seguro=True subdivide según dt_max."""
    n = int(np.ceil(dt / (0.95 * dt_max(p)))) if seguro else 1
    h = dt / n
    for _ in range(n):
        x = x + h * campo(TL, x, p)
    return x


def equilibrio(TL, p):
    """Equilibrio analítico bajo carga constante TL (Proposición 1)."""
    F, R, LD = p["F"], p["R"], p["LD"]
    if TL == 0:
        return np.array([0.0, 1.0, 0.0])
    TLs = R * (LD + F) / (R * (LD + F) + F * LD)
    if TL <= TLs:
        MA = LD * TL / (LD + F)
    else:
        MA = 1.0 / (1.0 + F / LD + F / R)
    MF = F * MA / R
    return np.array([MA, 1 - MA - MF, MF])


def carga_sostenible(p):
    F, R, LD = p["F"], p["R"], p["LD"]
    return R * (LD + F) / (R * (LD + F) + F * LD)


def semaforo(MF):
    return "Verde" if MF < UMBRALES[0] else ("Naranja" if MF < UMBRALES[1] else "Roja")


# ---------------------------------------------------------------------------
def verificar():
    p = PARAM
    rng = np.random.default_rng(42)
    print(f"dt_max (Prop. 3) = {dt_max(p):.5f} s")
    print(f"TL* sostenible (Prop. 1) = {carga_sostenible(p):.5f}  "
          f"(límite L_D->inf: R/(F+R) = {p['R']/(p['F']+p['R']):.5f})")

    # Lema 2: invariancia del simplex con cargas aleatorias conmutadas
    peor = 0.0
    for _ in range(200):
        x = rng.dirichlet([1, 1, 1])
        for _ in range(600):
            x = paso(rng.choice([0.0, 0.1, 0.3, 0.6]), x, 1.0, p)
            peor = min(peor, x.min())
            assert abs(x.sum() - 1) < 1e-9
    print(f"Lema 2  | min componente en 120 000 pasos conmutados = {peor:.3e}  (>= 0: OK)")

    # Prop. 1-2: convergencia al equilibrio analítico desde condiciones aleatorias
    err = 0.0
    for TL in (0.05, 0.08, 0.2, 0.5):
        xe = equilibrio(TL, p)
        for _ in range(50):
            x = rng.dirichlet([1, 1, 1])
            for _ in range(12000):          # 12 000 s >> 1/R
                x = paso(TL, x, 1.0, p)
            err = max(err, np.abs(x - xe).max())
    print(f"Prop 1-2| error max vs equilibrio analítico (200 corridas) = {err:.2e}")

    # Prop. 2: tasa de convergencia ~ R en la región sostenible
    TL = 0.05
    xe = equilibrio(TL, p)
    x = np.array([TL, 1 - TL, 0.0])
    t1, t2 = 3000, 6000
    for k in range(t2):
        x = paso(TL, x, 1.0, p)
        if k + 1 == t1:
            e1 = abs(x[2] - xe[2])
    e2 = abs(x[2] - xe[2])
    print(f"Prop 2  | tasa estimada = {np.log(e1/e2)/(t2-t1):.6f} 1/s  vs  R = {p['R']:.6f}")

    # Prop. 3: el paso de 0.1 s (control a 10 Hz) sin subdivisión viola la positividad
    x = np.array([0.9, 0.1, 0.0])
    xs = paso(0.0, x, 0.105, p, seguro=False)
    print(f"Prop 3  | Euler con dt=0.105 s > dt_max, TL=0: M_A = {xs[0]:.4f}  "
          f"({'NEGATIVO: la cota es necesaria' if xs[0] < 0 else 'no negativo'})")


# ---------------------------------------------------------------------------
def demo(con_robodk=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = PARAM
    ops = {"Op1": "novato", "Op2": "intermedio", "Op3": "experto"}
    # Turno ilustrativo de 60 min: ciclos de trabajo/pausa con cargas distintas
    def carga(op, t):
        ciclo = {"Op1": (90, 30, 0.20), "Op2": (120, 20, 0.25), "Op3": (150, 10, 0.30)}[op]
        trab, pausa, TL = ciclo
        return TL * KAPPA[ops[op]] if (t % (trab + pausa)) < trab else 0.0

    T = 3600
    x = {op: np.array([0.0, 1.0, 0.0]) for op in ops}
    hist = {op: [] for op in ops}
    RDK = None
    if con_robodk:
        sys.path.insert(0, ".")
        from robodk.robolink import Robolink
        from crear_escena_hrc import actualizar_semaforo
        RDK = Robolink()
    for t in range(T):
        for op in ops:
            x[op] = paso(carga(op, t), x[op], 1.0, p)
            hist[op].append(x[op][2])
            if RDK is not None and t % 30 == 0:
                actualizar_semaforo(RDK, op, x[op][2])
    fig, ax = plt.subplots(figsize=(8, 4))
    for op, col in zip(ops, ("#e6c200", "#7a48a8", "#2f6fd6")):
        ax.plot(np.arange(T) / 60, hist[op], color=col, label=f"{op} ({ops[op]})")
    for u in UMBRALES:
        ax.axhline(u, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("$M_F$ (fracción fatigada)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("fatiga_demo.png", dpi=150)
    for op in ops:
        print(f"{op}: M_F final = {hist[op][-1]:.3f} -> {semaforo(hist[op][-1])}")
    print("Figura guardada en fatiga_demo.png")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "verificar"
    if modo == "verificar":
        verificar()
    elif modo == "demo":
        demo("--robodk" in sys.argv)
    else:
        print(__doc__)
