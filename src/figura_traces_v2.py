# -*- coding: utf-8 -*-
"""
figura_traces_v2.py -- Redesigned Figure 5 (Reviewer 2, pt. 4).

The original figure plotted three densely oscillating traces per panel, which
saturated into a solid block and showed neither the margin to the threshold nor
the fact that the guarantee holds across seeds.  The redesign makes three
changes: traces are drawn as 2-min min/max envelopes, so the extremes that the
safety claim concerns stay visible; the saturated equilibrium of Proposition 1
is drawn as an explicit reference level; and a third panel reports the per-shift
maximum over all 30 seeds, which is where the pathwise guarantee can be read.

Release 1.1.1 (final layout for publication): the labels of the reference
levels sit in an empty area of panel (c), the shift-percentage boxes sit above
the data, the y-axis extends to 1.08 to make room for them, and the legend
lines are thicker and clear of the axis labels.  Data and values unchanged.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escudo_experimento import (LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
                                KAPPA, SPEED, RB_MODE, INTERARRIVAL, NSUB,
                                T_SHIFT, N_SEEDS, ESTACIONES, THETA, simular)
from estilo_figuras import aplicar

aplicar()
SAT = 0.9154                       # saturated equilibrium, Proposition 1
BIN = 120                          # envelope bin, seconds
COLS = ("#b58900", "#6c3fa0", "#2f6fd6")
LABS = ("Op1 (novice)", "Op2 (intermediate)", "Op3 (expert)")

fig = plt.figure(figsize=(6.8, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.62], wspace=0.34)
ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

rng = np.random.default_rng(1000)
arr = rng.poisson(1.0 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
tb = (np.arange(T_SHIFT // BIN) * BIN + BIN / 2) / 3600.0

for ax, sh, tit in ((ax0, False, "Greedy throughput"),
                    (ax1, True, "Greedy throughput + shield")):
    tr = np.zeros((T_SHIFT, 3))
    simular(1, sh, arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
            KAPPA, SPEED, RB_MODE, NSUB, tr)
    ax.axhspan(THETA, 1.08, color="#d64545", alpha=0.07, lw=0)
    for o in range(3):
        b = tr[:, o].reshape(-1, BIN)
        ax.fill_between(tb, b.min(1), b.max(1), color=COLS[o], alpha=0.42, lw=0)
        ax.plot(tb, b.max(1), lw=0.55, color=COLS[o], label=LABS[o])
    ax.axhline(THETA, ls="--", lw=0.9, color="k")
    ax.axhline(SAT, ls=":", lw=1.0, color="#8b0000")
    frac = 100.0 * np.mean(tr.max(axis=1) > THETA)
    ax.text(0.5, 0.99, f"{frac:.0f}% of shift above " + r"$\theta$",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
            bbox=dict(fc="white", ec="0.75", lw=0.5, pad=1.6))
    ax.set_title(tit, fontsize=9)
    ax.set_xlabel("Time (h)")
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

ax0.set_ylabel(r"$M_F$")
ax1.set_yticklabels([])

# ---- panel (c): per-shift maximum across all seeds
mx = {False: [], True: []}
for sh in (False, True):
    for seed in range(N_SEEDS):
        r2 = np.random.default_rng(1000 + seed)
        a2 = r2.poisson(1.0 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
        _, maxmf, *_ = simular(1, sh, a2, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR,
                               ROBOT, KAPPA, SPEED, RB_MODE, NSUB,
                               np.zeros((T_SHIFT, 3)))
        mx[sh].append(maxmf.max())

ax2.axhspan(THETA, 1.08, color="#d64545", alpha=0.07, lw=0)
jit = np.random.default_rng(7).uniform(-0.11, 0.11, N_SEEDS)
for i, (sh, c) in enumerate(((False, "#c0392b"), (True, "#1f4e8c"))):
    ax2.scatter(i + jit, mx[sh], s=9, color=c, alpha=0.8, lw=0, zorder=3)
    ax2.hlines(np.mean(mx[sh]), i - 0.26, i + 0.26, color=c, lw=1.4, zorder=4)
ax2.axhline(THETA, ls="--", lw=0.9, color="k")
ax2.axhline(SAT, ls=":", lw=1.0, color="#8b0000")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["No\nshield", "Shield"], fontsize=7.8)
ax2.set_xlim(-0.55, 1.55); ax2.set_ylim(0, 1.08)
ax2.set_yticks(np.arange(0, 1.01, 0.2)); ax2.set_yticklabels([])
ax2.set_title(f"All {N_SEEDS} seeds", fontsize=9)
ax2.annotate(f"Worst {max(mx[True]):.4f}", xy=(1.12, max(mx[True])),
             xytext=(1.50, 0.50), fontsize=6.9, ha="right", color="#1f4e8c",
             arrowprops=dict(arrowstyle="-", lw=0.55, color="#1f4e8c"))

# reference levels labelled where no data are drawn (panel c)
ax2.text(1.52, SAT + 0.012, r"$M_F^\star=0.9154$", ha="right", va="bottom",
         fontsize=7, color="#8b0000")
ax2.text(-0.50, THETA + 0.012, r"$\theta=0.40$", ha="left", va="bottom",
         fontsize=7)

h, l = ax0.get_legend_handles_labels()
leg = fig.legend(h, l, frameon=False, fontsize=7.6, loc="lower center",
                 ncol=3, bbox_to_anchor=(0.5, -0.17), handlelength=1.6)
for lh in leg.get_lines():
    lh.set_linewidth(1.8)

fig.savefig("fig_shield_traces.pdf", bbox_inches="tight")
fig.savefig("fig_shield_traces.png", bbox_inches="tight", dpi=300)
print(f"no shield: worst {max(mx[False]):.4f}  mean {np.mean(mx[False]):.4f}")
print(f"shield   : worst {max(mx[True]):.4f}  mean {np.mean(mx[True]):.4f}")
