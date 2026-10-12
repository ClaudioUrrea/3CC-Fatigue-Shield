# 3CC-Fatigue-Shield

Code, results and figures for the article

> C. Urrea, **"Sustainable Work Allocation Under the Three-Compartment Muscle Fatigue Model: Invariance, Closed-Form Rest Allowances, and a Certified Shield for Human–Robot Collaboration"**, submitted to *Mathematics* (MDPI), 2026.

Archived release: https://doi.org/10.6084/m9.figshare.33960379 · License: MIT

## What is here

| Path | Content |
|---|---|
| `src/fatiga.py` | 3CC-r fatigue model (Xia & Frey-Law 2008; Looft et al. 2018), simplex-preserving integrator, closed-form equilibrium and sustainable load. `python src/fatiga.py verificar` reproduces the verification table |
| `src/verificar_periodico.py` | Closed-form periodic peak vs. full model and minimum rest allowance checks |
| `src/figuras.py`, `src/estilo_figuras.py` | Figures of the periodic regime, rest-allowance curves and Euler bound (Palatino / TeX Gyre Pagella) |
| `src/escudo_experimento.py` | Shield experiment: 3 allocators × with/without shield × 2 demand levels × 30 eight-hour shifts |
| `results/resultados_escudo.json` | Raw results of the shield experiment |
| `robodk/crear_escena_hrc.py`, `robodk/visibilidad.py` | Programmatic construction of the three-operator, two-robot benchmark cell in RoboDK, with automatic reachability and collision checks, and reproducible camera views |
| `figures/` | All figures of the article (PDF and PNG) |

## Parameters (shoulder)

F = 0.01820 s⁻¹, R = 0.00168 s⁻¹ (Frey-Law et al. 2012, Table 1); r = 15 (general value of Looft et al. 2018; no shoulder data were available there); L_D = L_R = 10 (Xia & Frey-Law 2008).

## Reproducing the results

```bash
pip install -r requirements.txt
cd src
python fatiga.py verificar          # analytical results, seed 42
python verificar_periodico.py       # periodic peak and rest allowance table
python figuras.py                   # Figures 1-3
python escudo_experimento.py        # shield table and Figures 5-6 (seeds 1000-1029)
```

The RoboDK cell (optional) requires RoboDK and `pip install robodk`; open an empty station and run `python robodk/crear_escena_hrc.py`, then `python robodk/visibilidad.py mixto`. The Doosan H2017 and A0509 robot files must be downloaded from the RoboDK online library (paths are set at the top of the script).

## Citation

See `CITATION.cff`. Please cite the article and the archived release.
