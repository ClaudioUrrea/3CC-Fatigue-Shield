# -*- coding: utf-8 -*-
"""
escudo_experimento.py - Experimento del escudo de fatiga (Sección 9.3 del paper).

Tres asignadores de referencia (reglas fijas, voraz por throughput y reasignación
reactiva por umbral) se ejecutan con y sin el escudo sobre la celda de tres
operadores y dos robots, con la dinámica 3CC-r COMPLETA integrada con subpasos
que preservan el simplex. 30 semillas, turno de 8 h.

Requiere: numpy, numba, matplotlib.  Uso:  python escudo_experimento.py
"""
import json
import math
import numpy as np
from numba import njit

from fatiga import PARAM, dt_max

# ---------------------------------------------------------------- parámetros
F, R, LD, LR, RMULT = PARAM["F"], PARAM["R"], PARAM["LD"], PARAM["LR"], PARAM["r"]
THETA, THETA_ORANGE, EPS = 0.40, 0.25, 0.01       # umbral rojo, naranja, margen
BETA, COLLAB_TIME = 0.5, 0.8                      # colaboración: carga y duración
T_SHIFT, N_SEEDS = 8 * 3600, 30

# Estaciones: Interface, Replenishing, Transport, Picking, Sorting
ESTACIONES = ["Interface", "Replenishing", "Transport", "Picking", "Sorting"]
LOAD = np.array([0.22, 0.18, 0.35, 0.15, 0.12])       # carga nominal (fracción MVC)
TAU_H = np.array([40.0, 35.0, 50.0, 25.0, 25.0])      # duración humana nominal (s)
ROBOT_FACTOR = np.array([1.5, 1.0, 0.9, 1.3, 1.6])     # duración robot / humana
INTERARRIVAL = np.array([60.0, 70.0, 80.0, 45.0, 45.0])  # media de Poisson (s)
OPERATOR = np.array([0, 2, 2, 1, 1])                   # Op1, Op3, Op3, Op2, Op2
ROBOT = np.array([0, 0, 0, 1, 1])                      # H2017, A0509
KAPPA = np.array([1.20, 1.00, 0.85])                   # novato, intermedio, experto
SPEED = np.array([1.20, 1.00, 0.90])                   # factor de duración por habilidad
RB_MODE = np.array([2, 1, 1, 0, 0])                    # reglas: 0=H, 1=R, 2=C
POLICIES = ["Rule-based", "Greedy throughput", "Reactive threshold"]
NSUB = int(math.ceil(1.0 / (0.95 * dt_max(PARAM))))


@njit(cache=True)
def paso_3cc(ma, mr, mf, TL, nsub):
    h = 1.0 / nsub
    Reff = R * (RMULT if TL == 0.0 else 1.0)
    for _ in range(nsub):
        if ma < TL:
            C = LD * (TL - ma) if mr > TL - ma else LD * mr
        else:
            C = LR * (TL - ma)
        dma = C - F * ma
        dmr = -C + Reff * mf
        dmf = F * ma - Reff * mf
        ma += h * dma; mr += h * dmr; mf += h * dmf
    return ma, mr, mf


@njit(cache=True)
def psi(m, lam, tau):
    e = math.exp(-R * tau)
    return m * e + F * lam / R * (1.0 - e)


@njit(cache=True)
def simular(policy, shielded, arrivals, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
            KAPPA, SPEED, RB_MODE, nsub, traza):
    T = arrivals.shape[0]
    ns = LOAD.shape[0]
    queue = np.zeros(ns, np.int64)
    op_busy = np.zeros(3); rob_busy = np.zeros(2); op_load = np.zeros(3)
    ma = np.zeros(3); mr = np.ones(3); mf = np.zeros(3)
    maxmf = np.zeros(3); summf = np.zeros(3); above = 0
    done = 0; proposals = 0; overrides = 0; qsum = 0.0
    fin_t = np.full(4096, 1e18); fin_n = 0   # tiempos de término de tareas en curso
    for t in range(T):
        for s in range(ns):
            queue[s] += arrivals[t, s]
        for o in range(3):
            if op_busy[o] <= t:
                op_load[o] = 0.0
        for k in range(fin_n):
            if fin_t[k] <= t:
                done += 1; fin_t[k] = 1e18
        for s in range(ns):
            if queue[s] == 0:
                continue
            o = OPERATOR[s]; rb = ROBOT[s]
            hf = op_busy[o] <= t; rf = rob_busy[rb] <= t
            dh = TAU_H[s] * SPEED[o]; dr = TAU_H[s] * ROBOT_FACTOR[s]; dc = COLLAB_TIME * dh
            # --- propuesta del asignador
            mode = -1
            if policy == 0:
                m0 = RB_MODE[s]
                if (m0 == 0 and hf) or (m0 == 1 and rf) or (m0 == 2 and hf and rf):
                    mode = m0
            elif policy == 1:
                best = 1e18
                if hf and dh < best: best = dh; mode = 0
                if rf and dr < best: best = dr; mode = 1
                if hf and rf and dc < best: best = dc; mode = 2
            else:
                if mf[o] < 0.25:
                    if hf: mode = 0
                    elif rf: mode = 1
                else:
                    if rf: mode = 1
            if mode == -1:
                continue
            orig = mode
            # --- escudo
            if shielded and mode != 1:
                lam = KAPPA[o] * LOAD[s] * (BETA if mode == 2 else 1.0)
                dur = dc if mode == 2 else dh
                ok = max(mf[o], psi(mf[o], lam, dur)) <= 0.40 - 0.01
                if not ok:
                    lamc = KAPPA[o] * LOAD[s] * BETA
                    if mode == 0 and hf and rf and max(mf[o], psi(mf[o], lamc, dc)) <= 0.39:
                        mode = 2
                    elif rf:
                        mode = 1
                    else:
                        mode = -1          # se difiere
                if mode == -1:
                    continue
            # --- ejecución
            if mode == 0:
                op_busy[o] = t + dh; op_load[o] = KAPPA[o] * LOAD[s]; d = dh
            elif mode == 1:
                rob_busy[rb] = t + dr; d = dr
            else:
                op_busy[o] = t + dc; rob_busy[rb] = t + dc
                op_load[o] = KAPPA[o] * LOAD[s] * BETA; d = dc
            if orig != 1:
                proposals += 1
                if mode != orig:
                    overrides += 1
            queue[s] -= 1
            fin_t[fin_n % 4096] = t + d; fin_n = min(fin_n + 1, 4096)
        for o in range(3):
            ma[o], mr[o], mf[o] = paso_3cc(ma[o], mr[o], mf[o], op_load[o], nsub)
            if mf[o] > maxmf[o]: maxmf[o] = mf[o]
            summf[o] += mf[o]
            traza[t, o] = mf[o]
        if mf.max() > 0.40:
            above += 1
        qsum += queue.sum()
    return done, maxmf, summf / T, above / T, proposals, overrides, qsum / T


def correr(demanda=1.0):
    res = {}
    for p, nombre in enumerate(POLICIES):
        for sh in (False, True):
            filas = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(1000 + seed)
                arr = rng.poisson(demanda / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
                done, maxmf, meanmf, above, prop, over, qm = simular(
                    p, sh, arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
                    KAPPA, SPEED, RB_MODE, NSUB, np.zeros((T_SHIFT, 3)))
                jain = meanmf.sum() ** 2 / (3 * (meanmf ** 2).sum())
                filas.append([done / 8.0, maxmf.max(), above * 100, jain,
                              100 * over / prop if prop else 0.0, qm])
            a = np.array(filas)
            m, sd = a.mean(0), a.std(0, ddof=1)
            ci = 2.045 * sd / np.sqrt(N_SEEDS)
            res[f"{nombre}{' + shield' if sh else ''}"] = {
                "throughput_per_h": [m[0], ci[0]], "max_MF": [m[1], ci[1]],
                "max_MF_worst_seed": float(a[:, 1].max()),
                "pct_time_above_theta": [m[2], ci[2]], "jain": [m[3], ci[3]],
                "intervention_rate_pct": [m[4], ci[4]], "mean_queue": [m[5], ci[5]]}
    return res


def figuras(todo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from estilo_figuras import aplicar
    aplicar()
    # --- Figura: trayectorias de una semilla, voraz sin/con escudo
    rng = np.random.default_rng(1000)
    arr = rng.poisson(1.0 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    cols = ("#b58900", "#6c3fa0", "#2f6fd6")
    for ax, sh, tit in ((axs[0], False, "Greedy throughput"), (axs[1], True, "Greedy throughput + shield")):
        tr = np.zeros((T_SHIFT, 3))
        simular(1, sh, arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT, KAPPA, SPEED, RB_MODE, NSUB, tr)
        t = np.arange(T_SHIFT) / 3600
        for o, lab in enumerate(("Op1 (novice)", "Op2 (intermediate)", "Op3 (expert)")):
            ax.plot(t[::20], tr[::20, o], lw=0.8, color=cols[o], label=lab)
        ax.axhline(THETA, ls="--", lw=0.8, color="k")
        ax.set_title(tit); ax.set_xlabel("Time (h)")
    axs[0].set_ylabel(r"$M_F$"); axs[1].legend(frameon=False, loc="upper right")
    fig.savefig("fig_shield_traces.pdf"); fig.savefig("fig_shield_traces.png")
    # --- Figura: throughput y MF máximo por política
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    abrev = ("RB", "GT", "TH"); x = np.arange(3); w = 0.2
    for i, (esc, hatch) in enumerate((("nominal", ""), ("high_demand", "//"))):
        r = todo[esc]
        for j, sh in enumerate(("", " + shield")):
            thr = [r[n + sh]["throughput_per_h"][0] for n in POLICIES]
            mx = [r[n + sh]["max_MF"][0] for n in POLICIES]
            off = (2 * i + j - 1.5) * w
            c = "#c0392b" if sh == "" else "#1f4e8c"
            lab = f"{'no shield' if sh == '' else 'shield'}, {'nominal' if i == 0 else 'high demand'}"
            axs[0].bar(x + off, thr, w, color=c, hatch=hatch, edgecolor="white", label=lab)
            axs[1].bar(x + off, mx, w, color=c, hatch=hatch, edgecolor="white")
    axs[1].axhline(THETA, ls="--", lw=0.8, color="k")
    for ax, yl in ((axs[0], "Throughput (tasks/h)"), (axs[1], r"Maximum $M_F$")):
        ax.set_xticks(x); ax.set_xticklabels(abrev); ax.set_ylabel(yl)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06))
    fig.savefig("fig_shield_summary.pdf"); fig.savefig("fig_shield_summary.png")


if __name__ == "__main__":
    todo = {}
    for nombre, dem in (("nominal", 1.0), ("high_demand", 1.5)):
        r = correr(dem)
        todo[nombre] = r
        print(f"=== demanda {nombre} (x{dem}) ===")
        for k, v in r.items():
            print(f"{k:30s} thr={v['throughput_per_h'][0]:6.1f}±{v['throughput_per_h'][1]:.1f} "
                  f"q={v['mean_queue'][0]:6.1f} maxMF={v['max_MF'][0]:.3f} "
                  f"(peor {v['max_MF_worst_seed']:.3f}) >θ={v['pct_time_above_theta'][0]:5.1f}% "
                  f"J={v['jain'][0]:.3f} interv={v['intervention_rate_pct'][0]:.1f}%")
    with open("resultados_escudo.json", "w") as fh:
        json.dump(todo, fh, indent=2)
    figuras(todo)
