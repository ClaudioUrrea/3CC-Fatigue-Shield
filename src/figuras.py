# -*- coding: utf-8 -*-
"""Genera las figuras del manuscrito (PDF vectorial + PNG 600 dpi) con Palatino."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from estilo_figuras import aplicar
from fatiga import PARAM, paso, dt_max

fuente = aplicar()
p = PARAM
F, R, r = p["F"], p["R"], p["r"]


def pico_reducido(TL, Tw, Tr):
    return F * TL / R * (1 - np.exp(-R * Tw)) / (1 - np.exp(-R * (Tw + r * Tr)))


def descanso_min(TL, Tw, th):
    a = 1 - (F * TL / (R * th)) * (1 - np.exp(-R * Tw))
    if a <= 0:
        return np.nan
    return max(0.0, (-np.log(a) - R * Tw) / (r * R))


# ---- Figura 1: órbita periódica, modelo completo vs reducido -------------
TL, Tw, Tr, ncic = 0.20, 90, 45, 14
x = np.array([0.0, 1.0, 0.0]); t = 0; ts, mf = [], []
for _ in range(ncic):
    for fase, dur in ((TL, Tw), (0.0, Tr)):
        for _ in range(dur):
            x = paso(fase, x, 1.0, p); t += 1; ts.append(t / 60); mf.append(x[2])
m = 0.0; tr, mr = [], []; t = 0
for _ in range(ncic):
    for carga, dur, rate in ((TL, Tw, R), (0.0, Tr, r * R)):
        obj = F * carga / R if carga > 0 else 0.0
        for _ in range(dur):
            m = obj + (m - obj) * np.exp(-rate); t += 1; tr.append(t / 60); mr.append(m)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot(ts, mf, lw=1.2, color="#1f4e8c", label="Full 3CC-r model")
ax.plot(tr, mr, lw=1.0, ls="--", color="#c0392b", label="Reduced model")
ax.axhline(pico_reducido(TL, Tw, Tr), color="gray", lw=0.8, ls=":",
           label=r"Closed-form peak $\bar M_F$")
ax.set_xlabel("Time (min)"); ax.set_ylabel(r"$M_F$")
ax.legend(frameon=False, loc="lower right"); fig.savefig("fig_periodic.pdf")
fig.savefig("fig_periodic.png")

# ---- Figura 2: curvas de descanso mínimo (tipo Rohmert) ------------------
th = 0.40
loads = np.linspace(0.05, 0.6, 200)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for Tw_, ls in ((30, "-"), (60, "--"), (120, "-."), (240, ":")):
    ax.plot(loads * 100, [descanso_min(L, Tw_, th) for L in loads], ls=ls, color="k",
            lw=1.1, label=fr"$T_w={Tw_}$ s")
ax.set_xlabel(r"Target load $T_L$ (% MVC)"); ax.set_ylabel(r"Minimum rest $T_r^{\min}$ (s)")
ax.set_ylim(0, 400); ax.legend(frameon=False, title=fr"$\theta={th}$")
fig.savefig("fig_rest_allowance.pdf"); fig.savefig("fig_rest_allowance.png")

# ---- Figura 3: positividad de Euler según el paso ------------------------
hmax = dt_max(p)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.axhspan(-0.2, 0.0, color="#c0392b", alpha=0.07, lw=0)
peor = None
for h, col, lab in ((0.95 * hmax, "#1f4e8c", r"$h=0.95\,h_{\max}$"),
                    (1.10 * hmax, "#c0392b", r"$h=1.10\,h_{\max}$")):
    x = np.array([0.9, 0.1, 0.0]); tt, ma = [0.0], [x[0]]
    for k in range(6):
        x = paso(0.0, x, h, p, seguro=False); tt.append((k + 1) * h); ma.append(x[0])
    ax.plot(tt, ma, marker="o", ms=3.2, lw=1.1, color=col, label=lab)
    if peor is None or min(ma) < peor[1]:
        peor = (tt[int(np.argmin(ma))], min(ma))
ax.axhline(0, color="k", lw=0.7)
# el primer paso por encima de la cota deja M_A fuera del simplex
ax.annotate(f"$M_A={peor[1]:.3f}$".replace("-", "−"),
            xy=peor, xytext=(peor[0] + 0.13, peor[1] - 0.07),
            fontsize=8, color="#c0392b", ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#c0392b"))
ax.set_xlabel("Time (s)"); ax.set_ylabel(r"$M_A$ (Explicit Euler, rest)")
ax.set_xlim(-0.02, 0.70); ax.set_ylim(-0.22, 0.98)
ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
ax.legend(frameon=False, loc="upper right")
fig.savefig("fig_euler.pdf"); fig.savefig("fig_euler.png")
print("Fuente usada:", fuente)
print("Pico cerrado (Fig.1):", round(pico_reducido(TL, Tw, Tr), 4), " pico simulado último ciclo:",
      round(max(mf[-(Tw + Tr):]), 4))
