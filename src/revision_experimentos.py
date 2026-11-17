# -*- coding: utf-8 -*-
"""
revision_experimentos.py -- Experiments added in revision round 1.

E0  Explicit bound on the reduced-model error epsilon           (Reviewer 1, pt. 1)
E1  Tightness of the step-size bound in all three branches      (Reviewer 1, pt. 3)
E2  Minimax-fair reassignment among several admissible operators(Reviewer 1, pt. 4)
E3  Sensitivity of the rest allowance to the multiplier r       (Reviewer 1, pt. 5)
E4  Robustness to inter-individual parameter variation          (Reviewer 3, pt. 2)
E5  Threshold sweep: safety / throughput trade-off              (Reviewer 3, pt. 5)
E6  Paired statistics on common random numbers                  (Reviewer 3, pt. 6)

Run:  python revision_experimentos.py
"""
import json, math
import numpy as np
from numba import njit
from scipy import stats

from fatiga import PARAM, dt_max, equilibrio
from escudo_experimento import (F, R, LD, LR, RMULT, BETA, COLLAB_TIME,
                                T_SHIFT, N_SEEDS, LOAD, TAU_H, ROBOT_FACTOR,
                                INTERARRIVAL, OPERATOR, ROBOT, KAPPA, SPEED,
                                RB_MODE, NSUB, ESTACIONES, simular, psi)

OUT = {}
rule = lambda s: print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


# =====================================================================
# E0. Explicit bound on the reduced-model approximation error
# =====================================================================
def eps_bound(TLmax, p=PARAM):
    """Three-term bound on sup_t |Mhat_F - M_F| derived in the revision."""
    F_, R_, LD_, LR_, r_ = p["F"], p["R"], p["LD"], p["LR"], p["r"]
    t1 = F_ ** 2 * TLmax / (R_ * (LD_ + F_))          # persistent recruitment offset
    t2 = F_ * TLmax / (LD_ + F_ - R_)                 # work-phase boundary layer
    t3 = F_ * TLmax / (LR_ + F_ - r_ * R_)            # rest-phase boundary layer
    return t1 + t2 + t3, (t1, t2, t3)


def reducido_vs_completo(TL, Tw, Tr, ncyc=80, p=PARAM):
    """Peak |error| between reduced and full 3CC-r over ncyc duty cycles."""
    F_, R_, r_ = p["F"], p["R"], p["r"]
    x = np.array([0.0, 1.0, 0.0]); mhat = 0.0; worst = 0.0
    dtmax = 0.95 * dt_max(p);
    for _ in range(ncyc):
        for TLc, dur in ((TL, Tw), (0.0, Tr)):
            n = int(math.ceil(dur / dtmax)); h = dur / n
            Reff = R_ * (r_ if TLc == 0.0 else 1.0)
            for _ in range(n):
                MA, MR, MF = x
                if MA < TLc:
                    C = p["LD"] * (TLc - MA) if MR > TLc - MA else p["LD"] * MR
                else:
                    C = p["LR"] * (TLc - MA)
                x = x + h * np.array([C - F_ * MA, -C + Reff * MF, F_ * MA - Reff * MF])
                mhat = mhat + h * (F_ * TLc - Reff * mhat)
                worst = max(worst, abs(mhat - x[2]))
    return worst


def E0():
    rule("E0  Explicit bound on the reduced-model error (Reviewer 1, pt. 1)")
    lam_max = float(max(KAPPA[OPERATOR[s]] * LOAD[s] for s in range(len(LOAD))))
    b, terms = eps_bound(lam_max)
    print(f"benchmark lambda_max = {lam_max:.4f}")
    print(f"bound terms: persistent={terms[0]:.3e}  work-layer={terms[1]:.3e}  "
          f"rest-layer={terms[2]:.3e}")
    print(f"=> epsilon_bound = {b:.5f}   (margin used in the paper: 0.01)")
    casos = [(0.10, 60, 30), (0.20, 90, 30), (0.30, 120, 60),
             (0.15, 300, 60), (0.2975, 50, 40), (0.264, 40, 30)]
    filas = []
    for TL, Tw, Tr in casos:
        obs = reducido_vs_completo(TL, Tw, Tr)
        bb, _ = eps_bound(TL)
        filas.append((TL, Tw, Tr, obs, bb))
        print(f"  TL={TL:.4f} Tw={Tw:4d} Tr={Tr:3d} | observed={obs:.3e}  "
              f"bound={bb:.3e}  ratio={obs/bb:.3f}  {'OK' if obs<=bb else 'VIOLATED'}")
    OUT["E0"] = {"lambda_max": lam_max, "eps_bound_benchmark": b,
                 "terms": terms, "cases": filas}


# =====================================================================
# E1. Tightness of the step-size bound in each of the three branches
# =====================================================================
def campo_np(TL, x, p=PARAM):
    MA, MR, MF = x
    Reff = p["R"] * (p["r"] if TL == 0 else 1.0)
    if MA < TL:
        C = p["LD"] * (TL - MA) if MR > TL - MA else p["LD"] * MR
    else:
        C = p["LR"] * (TL - MA)
    return np.array([C - p["F"] * MA, -C + Reff * MF, p["F"] * MA - Reff * MF])


def E1():
    rule("E1  Tightness of h_max in all three branches (Reviewer 1, pt. 3)")
    p = PARAM; res = {}
    # (a) L_R + F  binds: rest from a purely active state -> M_A < 0
    h = 1.0 / (p["LR"] + p["F"]) * 1.0001
    x = np.array([0.9, 0.1, 0.0]); xn = x + h * campo_np(0.0, x, p)
    res["LR+F"] = {"h": h, "M_A_next": xn[0]}
    print(f"(a) L_R+F = {p['LR']+p['F']:.4f} : h={h:.6f} -> M_A = {xn[0]:+.4e}")
    # (b) L_D binds: branch 1 with M_R just above T_L - M_A and M_F = 0 -> M_R < 0
    pb = dict(p, LD=40.0, LR=1.0)            # a joint where L_D is the binding term
    h = 1.0 / pb["LD"] * 1.0001
    TL = 0.5; MA = 0.2; MR = (TL - MA) * 1.0000001
    x = np.array([MA, MR, 1 - MA - MR]); xn = x + h * campo_np(TL, x, pb)
    res["LD"] = {"h": h, "M_R_next": xn[1], "LD": pb["LD"]}
    print(f"(b) L_D   = {pb['LD']:.4f} : h={h:.6f} -> M_R = {xn[1]:+.4e}")
    # (c) r R binds: a joint with fast rest recovery; rest from M_A = 0 -> M_F < 0
    pc = dict(p, LD=1.0, LR=1.0, R=0.05, r=25.0)   # r R = 1.25 > L_R + F
    h = 1.0 / (pc["r"] * pc["R"]) * 1.0001
    x = np.array([0.0, 0.4, 0.6]); xn = x + h * campo_np(0.0, x, pc)
    res["rR"] = {"h": h, "M_F_next": xn[2], "rR": pc["r"] * pc["R"]}
    print(f"(c) r R   = {pc['r']*pc['R']:.4f} : h={h:.6f} -> M_F = {xn[2]:+.4e}")
    OUT["E1"] = res


# =====================================================================
# E2. Minimax-fair reassignment, cross-trained cell
# =====================================================================
@njit(cache=True)
def sim_cross(rule_id, arrivals, LOAD, TAU_H, ROBOT_FACTOR, ROBOT, KAPPA, SPEED,
              nsub, theta, eps, seed):
    """Cross-trained cell: every station may be served by any of the 3 operators.

    rule_id 0 = minimax post-task margin  (Definition 2 of the paper)
            1 = first admissible operator (index order)
            2 = random admissible operator
            3 = smallest *current* M_F among admissible
            4 = smallest cumulative exposure among admissible (corrected rule)
    """
    np.random.seed(seed)
    T = arrivals.shape[0]; ns = LOAD.shape[0]
    queue = np.zeros(ns, np.int64)
    op_busy = np.zeros(3); rob_busy = np.zeros(2); op_load = np.zeros(3)
    ma = np.zeros(3); mr = np.ones(3); mf = np.zeros(3)
    maxmf = np.zeros(3); summf = np.zeros(3); cum = np.zeros(3)
    nsel = np.zeros(3)
    done = 0; contested = 0; onestep_ok = 0
    fin_t = np.full(4096, 1e18); fin_n = 0
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
            rb = ROBOT[s]; rf = rob_busy[rb] <= t
            # --- admissible operators for this task
            cand = np.full(3, -1.0)          # post-task margin, -1 = not admissible
            post_v = np.full(3, 2.0)
            nadm = 0
            for o in range(3):
                if op_busy[o] > t:
                    continue
                lam = KAPPA[o] * LOAD[s]; dur = TAU_H[s] * SPEED[o]
                post = psi(mf[o], lam, dur)
                if max(mf[o], post) <= theta - eps:
                    cand[o] = theta - eps - post
                    post_v[o] = post
                    nadm += 1
            if nadm > 1:
                contested += 1
            pick = -1
            if nadm > 0:
                if rule_id == 0:                      # largest post-task margin
                    best = -1.0
                    for o in range(3):
                        if cand[o] > best:
                            best = cand[o]; pick = o
                elif rule_id == 1:                    # first admissible
                    for o in range(3):
                        if cand[o] >= 0.0:
                            pick = o; break
                elif rule_id == 2:                    # random admissible
                    k = int(np.random.random() * nadm)
                    c = 0
                    for o in range(3):
                        if cand[o] >= 0.0:
                            if c == k:
                                pick = o; break
                            c += 1
                elif rule_id == 3:                    # smallest current M_F
                    best = 2.0
                    for o in range(3):
                        if cand[o] >= 0.0 and mf[o] < best:
                            best = mf[o]; pick = o
                else:                                 # smallest cumulative exposure
                    best = 1e18
                    for o in range(3):
                        if cand[o] >= 0.0 and cum[o] < best:
                            best = cum[o]; pick = o
            # --- does the choice minimise the post-decision maximum?
            if nadm > 1 and pick >= 0:
                others = 0.0
                for o in range(3):
                    if o != pick and mf[o] > others:
                        others = mf[o]
                achieved = max(post_v[pick], others)
                bestposs = 1e18
                for o in range(3):
                    if cand[o] >= 0.0:
                        oth = 0.0
                        for q in range(3):
                            if q != o and mf[q] > oth:
                                oth = mf[q]
                        v = max(post_v[o], oth)
                        if v < bestposs:
                            bestposs = v
                if achieved <= bestposs + 1e-12:
                    onestep_ok += 1
            if pick >= 0:
                d = TAU_H[s] * SPEED[pick]
                op_busy[pick] = t + d; op_load[pick] = KAPPA[pick] * LOAD[s]
                cum[pick] += KAPPA[pick] * LOAD[s] * d; nsel[pick] += 1
            elif rf:
                d = TAU_H[s] * ROBOT_FACTOR[s]; rob_busy[rb] = t + d
            else:
                continue
            queue[s] -= 1
            fin_t[fin_n % 4096] = t + d; fin_n = min(fin_n + 1, 4096)
        for o in range(3):
            ma[o], mr[o], mf[o] = paso_local(ma[o], mr[o], mf[o], op_load[o], nsub)
            if mf[o] > maxmf[o]:
                maxmf[o] = mf[o]
            summf[o] += mf[o]
    return done, maxmf, summf / T, contested, onestep_ok, nsel


@njit(cache=True)
def paso_local(ma, mr, mf, TL, nsub):
    h = 1.0 / nsub
    Reff = R * (RMULT if TL == 0.0 else 1.0)
    for _ in range(nsub):
        if ma < TL:
            C = LD * (TL - ma) if mr > TL - ma else LD * mr
        else:
            C = LR * (TL - ma)
        ma, mr, mf = (ma + h * (C - F * ma), mr + h * (-C + Reff * mf),
                      mf + h * (F * ma - Reff * mf))
    return ma, mr, mf


def E2(demanda=1.3):
    rule("E2  Minimax-fair reassignment, cross-trained cell (Reviewer 1, pt. 4)")
    nombres = ["Minimax margin (Def. 2)", "First admissible",
               "Random admissible", "Smallest current M_F",
               "Smallest cumulative exposure"]
    res = {}
    for rid, nom in enumerate(nombres):
        filas = []; sel = np.zeros(3)
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(2000 + seed)
            arr = rng.poisson(demanda / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
            done, maxmf, meanmf, cont, ok1, nsel = sim_cross(
                rid, arr, LOAD, TAU_H, ROBOT_FACTOR, ROBOT, KAPPA, SPEED,
                NSUB, 0.40, 0.01, 3000 + seed)
            jain = meanmf.sum() ** 2 / (3 * (meanmf ** 2).sum())
            filas.append([done / 8.0, maxmf.max(), jain,
                          meanmf.max() - meanmf.min(), cont,
                          100 * ok1 / cont if cont else 100.0])
            sel += nsel
        a = np.array(filas); m = a.mean(0); ci = 2.045 * a.std(0, ddof=1) / np.sqrt(N_SEEDS)
        share = 100 * sel / sel.sum()
        res[nom] = {"throughput": [m[0], ci[0]], "max_MF": [m[1], ci[1]],
                    "jain": [m[2], ci[2]], "spread": [m[3], ci[3]],
                    "contested": m[4], "onestep_optimal_pct": m[5],
                    "share_pct": share.tolist(),
                    "raw_max": a[:, 1].tolist(),
                    "raw_spread": a[:, 3].tolist(), "raw_jain": a[:, 2].tolist()}
        print(f"{nom:30s} thr={m[0]:6.1f} maxMF={m[1]:.4f} Jain={m[2]:.4f} "
              f"spread={m[3]:.4f} 1-step-opt={m[5]:5.1f}% "
              f"share=[{share[0]:4.1f} {share[1]:4.1f} {share[2]:4.1f}]%")
    base = res[nombres[0]]
    for nom in nombres[1:]:
        _, pv = stats.wilcoxon(base["raw_spread"], res[nom]["raw_spread"])
        print(f"  Wilcoxon spread  minimax vs {nom:30s}: p = {pv:.2e}")
        res[nom]["wilcoxon_spread_vs_minimax"] = pv
    OUT["E2"] = res


# =====================================================================
# E3. Sensitivity of the rest allowance to r
# =====================================================================
def Tr_min(TL, Tw, theta, p=PARAM):
    F_, R_, r_ = p["F"], p["R"], p["r"]
    a = 1.0 - F_ * TL / (R_ * theta) * (1.0 - math.exp(-R_ * Tw))
    if a <= 0:
        return math.inf
    return max(0.0, (-math.log(a) - R_ * Tw) / (r_ * R_))


def E3():
    rule("E3  Sensitivity of the rest allowance to r (Reviewer 1, pt. 5)")
    print("Closed form: T_r^min = [-ln a - R T_w]_+ / (r R) with a independent of r")
    print("=> elasticity d ln T_r^min / d ln r = -1 exactly; r rescales but never")
    print("   changes feasibility (the condition a > 0 does not involve r).")
    filas = []
    for (TL, Tw) in ((0.20, 90.0), (0.30, 60.0), (0.10, 120.0)):
        row = {"TL": TL, "Tw": Tw, "vals": []}
        for r_ in (10.0, 12.0, 15.0, 18.0, 20.0):
            v = Tr_min(TL, Tw, 0.40, dict(PARAM, r=r_))
            row["vals"].append((r_, v))
        base = dict(row["vals"])[15.0]
        s = "  ".join(f"r={r_:4.1f}:{v:6.1f}s ({v/base:4.2f}x)" for r_, v in row["vals"])
        print(f"  TL={TL:.2f} Tw={Tw:5.1f}s | {s}")
        filas.append(row)
    # the shield test psi does not involve r at all -> safety is r-independent
    print("\nShield admissibility test psi(m;lambda,tau) = m e^{-R tau} + (F lambda / R)(1 - e^{-R tau})")
    print("contains no r: the pathwise guarantee of Theorem 5 is invariant to r.")
    print("r affects only how fast operators recover between tasks, i.e. throughput:")
    res = {}
    for r_ in (5.0, 10.0, 15.0, 20.0, 25.0):
        filas_r = []
        nsub = int(math.ceil(1.0 / (0.95 * dt_max(dict(PARAM, r=r_)))))
        for seed in range(20):
            rng = np.random.default_rng(1000 + seed)
            arr = rng.poisson(1.5 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
            done, maxmf = sim_robusto(
                arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT, KAPPA, SPEED,
                nsub, 0.40, 0.01, np.full(3, F), np.full(3, R), np.full(3, r_))
            filas_r.append([done / 8.0, maxmf.max()])
        a = np.array(filas_r); m = a.mean(0)
        res[r_] = {"throughput": m[0], "max_MF": m[1]}
        print(f"  r={r_:4.1f} | throughput={m[0]:6.1f} tasks/h  max M_F={m[1]:.4f}  "
              f"{'SAFE' if m[1] <= 0.40 else 'VIOLATED'}")
    OUT["E3"] = {"Tr_min_table": filas, "shield_vs_r": res}


# =====================================================================
# E4. Robustness to inter-individual parameter variation
# =====================================================================
@njit(cache=True)
def paso_param(ma, mr, mf, TL, nsub, Fi, Ri, ri):
    h = 1.0 / nsub
    Reff = Ri * (ri if TL == 0.0 else 1.0)
    for _ in range(nsub):
        if ma < TL:
            C = LD * (TL - ma) if mr > TL - ma else LD * mr
        else:
            C = LR * (TL - ma)
        ma, mr, mf = (ma + h * (C - Fi * ma), mr + h * (-C + Reff * mf),
                      mf + h * (Fi * ma - Reff * mf))
    return ma, mr, mf


@njit(cache=True)
def psi_d(m, lam, tau, Fd, Rd):
    e = math.exp(-Rd * tau)
    return m * e + Fd * lam / Rd * (1.0 - e)


@njit(cache=True)
def sim_robusto2(arrivals, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT, KAPPA,
                 SPEED, nsub, theta, eps, Ftrue, Rtrue, rtrue, Fd, Rd):
    """Shield designed with (Fd,Rd); operators have true (Ftrue,Rtrue,rtrue)."""
    T = arrivals.shape[0]; ns = LOAD.shape[0]
    queue = np.zeros(ns, np.int64)
    op_busy = np.zeros(3); rob_busy = np.zeros(2); op_load = np.zeros(3)
    ma = np.zeros(3); mr = np.ones(3); mf = np.zeros(3)
    maxmf = np.zeros(3); done = 0
    fin_t = np.full(4096, 1e18); fin_n = 0
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
            dh = TAU_H[s] * SPEED[o]; dr = TAU_H[s] * ROBOT_FACTOR[s]
            dc = COLLAB_TIME * dh
            best = 1e18; mode = -1
            if hf and dh < best: best = dh; mode = 0
            if rf and dr < best: best = dr; mode = 1
            if hf and rf and dc < best: best = dc; mode = 2
            if mode == -1:
                continue
            if mode != 1:
                lam = KAPPA[o] * LOAD[s] * (BETA if mode == 2 else 1.0)
                dur = dc if mode == 2 else dh
                if max(mf[o], psi_d(mf[o], lam, dur, Fd, Rd)) > theta - eps:
                    lamc = KAPPA[o] * LOAD[s] * BETA
                    if mode == 0 and hf and rf and max(mf[o], psi_d(mf[o], lamc, dc, Fd, Rd)) <= theta - eps:
                        mode = 2
                    elif rf:
                        mode = 1
                    else:
                        continue
            if mode == 0:
                op_busy[o] = t + dh; op_load[o] = KAPPA[o] * LOAD[s]; d = dh
            elif mode == 1:
                rob_busy[rb] = t + dr; d = dr
            else:
                op_busy[o] = t + dc; rob_busy[rb] = t + dc
                op_load[o] = KAPPA[o] * LOAD[s] * BETA; d = dc
            queue[s] -= 1
            fin_t[fin_n % 4096] = t + d; fin_n = min(fin_n + 1, 4096)
        for o in range(3):
            ma[o], mr[o], mf[o] = paso_param(ma[o], mr[o], mf[o], op_load[o],
                                             nsub, Ftrue[o], Rtrue[o], rtrue[o])
            if mf[o] > maxmf[o]:
                maxmf[o] = mf[o]
    return done, maxmf


@njit(cache=True)
def sim_robusto(arrivals, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT, KAPPA,
                SPEED, nsub, theta, eps, Ftrue, Rtrue, rtrue):
    """Shield designed with nominal (F,R); operators have true (Ftrue,Rtrue)."""
    T = arrivals.shape[0]; ns = LOAD.shape[0]
    queue = np.zeros(ns, np.int64)
    op_busy = np.zeros(3); rob_busy = np.zeros(2); op_load = np.zeros(3)
    ma = np.zeros(3); mr = np.ones(3); mf = np.zeros(3)
    maxmf = np.zeros(3); done = 0
    fin_t = np.full(4096, 1e18); fin_n = 0
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
            dh = TAU_H[s] * SPEED[o]; dr = TAU_H[s] * ROBOT_FACTOR[s]
            dc = COLLAB_TIME * dh
            best = 1e18; mode = -1
            if hf and dh < best: best = dh; mode = 0
            if rf and dr < best: best = dr; mode = 1
            if hf and rf and dc < best: best = dc; mode = 2
            if mode == -1:
                continue
            if mode != 1:
                lam = KAPPA[o] * LOAD[s] * (BETA if mode == 2 else 1.0)
                dur = dc if mode == 2 else dh
                if max(mf[o], psi(mf[o], lam, dur)) > theta - eps:   # nominal test
                    lamc = KAPPA[o] * LOAD[s] * BETA
                    if mode == 0 and hf and rf and max(mf[o], psi(mf[o], lamc, dc)) <= theta - eps:
                        mode = 2
                    elif rf:
                        mode = 1
                    else:
                        continue
            if mode == 0:
                op_busy[o] = t + dh; op_load[o] = KAPPA[o] * LOAD[s]; d = dh
            elif mode == 1:
                rob_busy[rb] = t + dr; d = dr
            else:
                op_busy[o] = t + dc; rob_busy[rb] = t + dc
                op_load[o] = KAPPA[o] * LOAD[s] * BETA; d = dc
            queue[s] -= 1
            fin_t[fin_n % 4096] = t + d; fin_n = min(fin_n + 1, 4096)
        for o in range(3):
            ma[o], mr[o], mf[o] = paso_param(ma[o], mr[o], mf[o], op_load[o],
                                             nsub, Ftrue[o], Rtrue[o], rtrue[o])
            if mf[o] > maxmf[o]:
                maxmf[o] = mf[o]
    return done, maxmf


def E4():
    rule("E4  Robustness to inter-individual parameter variation (Reviewer 3, pt. 2)")
    print("Sufficient condition (Proposition 6): the guarantee is preserved whenever")
    print("F'/R' <= F/R and R' <= R, because psi is then pointwise non-increasing.")
    res = {}
    for cv in (0.10, 0.20, 0.30):
        viol = 0; runs = 0; worst = 0.0; ratio_viol = []
        for seed in range(120):
            rng = np.random.default_rng(7000 + seed)
            Ft = F * np.exp(rng.normal(0, cv, 3))
            Rt = R * np.exp(rng.normal(0, cv, 3))
            rt = np.full(3, RMULT)
            arr = rng.poisson(1.5 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
            _, maxmf = sim_robusto(arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
                                   KAPPA, SPEED, NSUB, 0.40, 0.01, Ft, Rt, rt)
            runs += 1
            if maxmf.max() > 0.40:
                viol += 1
                ratio_viol.append(float(np.max((Ft / Rt) / (F / R))))
            worst = max(worst, float(maxmf.max()))
        res[cv] = {"violation_pct": 100 * viol / runs, "worst_MF": worst,
                   "n": runs}
        print(f"  log-normal CV={cv:.0%} | theta exceeded in {100*viol/runs:5.1f}% of "
              f"{runs} shifts; worst M_F = {worst:.4f}")
        if ratio_viol:
            print(f"      violations had (F'/R')/(F/R) in "
                  f"[{min(ratio_viol):.2f}, {max(ratio_viol):.2f}]")
    # (a) inflating the margin is an inefficient remedy
    print("\n  (a) inflating the margin epsilon, nominal design parameters, CV=30%:")
    for eps in (0.01, 0.04, 0.08, 0.12):
        viol = 0; thr = []
        for seed in range(60):
            rng = np.random.default_rng(9000 + seed)
            Ft = F * np.exp(rng.normal(0, 0.30, 3)); Rt = R * np.exp(rng.normal(0, 0.30, 3))
            arr = rng.poisson(1.5 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
            d, mm = sim_robusto(arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
                                KAPPA, SPEED, NSUB, 0.40, eps, Ft, Rt, np.full(3, RMULT))
            thr.append(d / 8.0)
            if mm.max() > 0.40:
                viol += 1
        print(f"    eps={eps:.2f} -> {100*viol/60:5.1f}% of 60 shifts exceed theta "
              f"(throughput {np.mean(thr):6.1f})")
        res[f"eps_{eps}"] = 100 * viol / 60
    # (b) designing the shield with conservative percentile parameters
    print("\n  (b) conservative design parameters F_d=F e^{k s}, R_d=R e^{-k s}, CV=30%:")
    for k in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0):
        Fd = F * math.exp(k * 0.30); Rd = R * math.exp(-k * 0.30)
        viol = 0; thr = []; worst = 0.0
        for seed in range(60):
            rng = np.random.default_rng(9000 + seed)
            Ft = F * np.exp(rng.normal(0, 0.30, 3)); Rt = R * np.exp(rng.normal(0, 0.30, 3))
            arr = rng.poisson(1.5 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
            d, mm = sim_robusto2(arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR, ROBOT,
                                 KAPPA, SPEED, NSUB, 0.40, 0.01, Ft, Rt,
                                 np.full(3, RMULT), Fd, Rd)
            thr.append(d / 8.0); worst = max(worst, float(mm.max()))
            if mm.max() > 0.40:
                viol += 1
        print(f"    k={k:.1f} (F_d/R_d = {(Fd/Rd)/(F/R):5.2f}x nominal) -> "
              f"{100*viol/60:5.1f}% exceed theta, worst={worst:.4f}, "
              f"throughput={np.mean(thr):6.1f}")
        res[f"k_{k}"] = {"viol_pct": 100 * viol / 60, "worst": worst,
                         "throughput": float(np.mean(thr))}
    OUT["E4"] = res


# =====================================================================
# E5. Threshold sweep
# =====================================================================
def E5():
    rule("E5  Threshold sweep: safety / throughput trade-off (Reviewer 3, pt. 5)")
    import escudo_experimento as EE
    res = {}
    for theta in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        row = {}
        for p, nom in ((0, "RB"), (1, "GT"), (2, "TH")):
            filas = []
            for seed in range(15):
                rng = np.random.default_rng(1000 + seed)
                arr = rng.poisson(1.5 / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
                done, maxmf, meanmf, above, prop, over, qm = sim_theta(
                    p, arr, NSUB, theta, 0.01)
                filas.append([done / 8.0, maxmf.max(),
                              100 * over / prop if prop else 0.0])
            a = np.array(filas); m = a.mean(0)
            row[nom] = {"throughput": m[0], "max_MF": m[1], "interv": m[2]}
        res[theta] = row
        s = "  ".join(f"{n}: thr={row[n]['throughput']:6.1f} int={row[n]['interv']:4.1f}%"
                      for n in ("RB", "GT", "TH"))
        print(f"  theta={theta:.2f} | maxMF={row['GT']['max_MF']:.4f} | {s}")
    OUT["E5"] = res


@njit(cache=True)
def sim_theta(policy, arrivals, nsub, theta, eps):
    T = arrivals.shape[0]; ns = LOAD.shape[0]
    queue = np.zeros(ns, np.int64)
    op_busy = np.zeros(3); rob_busy = np.zeros(2); op_load = np.zeros(3)
    ma = np.zeros(3); mr = np.ones(3); mf = np.zeros(3)
    maxmf = np.zeros(3); summf = np.zeros(3); above = 0
    done = 0; proposals = 0; overrides = 0; qsum = 0.0
    fin_t = np.full(4096, 1e18); fin_n = 0
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
            dh = TAU_H[s] * SPEED[o]; dr = TAU_H[s] * ROBOT_FACTOR[s]
            dc = COLLAB_TIME * dh
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
            if mode != 1:
                lam = KAPPA[o] * LOAD[s] * (BETA if mode == 2 else 1.0)
                dur = dc if mode == 2 else dh
                if max(mf[o], psi(mf[o], lam, dur)) > theta - eps:
                    lamc = KAPPA[o] * LOAD[s] * BETA
                    if mode == 0 and hf and rf and max(mf[o], psi(mf[o], lamc, dc)) <= theta - eps:
                        mode = 2
                    elif rf:
                        mode = 1
                    else:
                        mode = -1
                if mode == -1:
                    continue
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
            ma[o], mr[o], mf[o] = paso_local(ma[o], mr[o], mf[o], op_load[o], nsub)
            if mf[o] > maxmf[o]: maxmf[o] = mf[o]
            summf[o] += mf[o]
        if mf.max() > theta: above += 1
        qsum += queue.sum()
    return done, maxmf, summf / T, above / T, proposals, overrides, qsum / T


# =====================================================================
# E6. Paired statistics on common random numbers
# =====================================================================
def E6():
    rule("E6  Paired statistics on common random numbers (Reviewer 3, pt. 6)")
    res = {}
    for dem, etiq in ((1.0, "nominal"), (1.5, "high")):
        for p, nom in ((0, "RB"), (1, "GT"), (2, "TH")):
            thr0 = []; thr1 = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(1000 + seed)
                arr = rng.poisson(dem / INTERARRIVAL, size=(T_SHIFT, len(ESTACIONES)))
                z = np.zeros((T_SHIFT, 3))
                d0 = simular(p, False, arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR,
                             ROBOT, KAPPA, SPEED, RB_MODE, NSUB, z)[0]
                d1 = simular(p, True, arr, LOAD, TAU_H, ROBOT_FACTOR, OPERATOR,
                             ROBOT, KAPPA, SPEED, RB_MODE, NSUB, z)[0]
                thr0.append(d0 / 8.0); thr1.append(d1 / 8.0)
            a, b = np.array(thr0), np.array(thr1)
            d = b - a
            tstat, pv = stats.ttest_rel(b, a)
            try:
                _, pw = stats.wilcoxon(b, a)
            except Exception:
                pw = float("nan")
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
            ci = stats.t.interval(0.95, len(d) - 1, d.mean(),
                                  d.std(ddof=1) / np.sqrt(len(d)))
            res[f"{etiq}_{nom}"] = {"mean_delta": d.mean(), "ci": list(ci),
                                    "p_ttest": pv, "p_wilcoxon": pw, "cohen_dz": dz,
                                    "pct": 100 * d.mean() / a.mean()}
            print(f"  {etiq:8s} {nom}: delta = {d.mean():+7.2f} tasks/h "
                  f"[{ci[0]:+.2f}, {ci[1]:+.2f}]  ({100*d.mean()/a.mean():+5.1f}%)  "
                  f"p_t={pv:.1e}  p_W={pw:.1e}  dz={dz:+.2f}")
    OUT["E6"] = res


# =====================================================================
if __name__ == "__main__":
    E0(); E1(); E2(); E3(); E4(); E5(); E6()
    with open("resultados_revision.json", "w") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    print("\nSaved -> resultados_revision.json")
