# 3CC-Fatigue-Shield

Code, results and figures for the article

> C. Urrea, **"Sustainable Work Allocation Under the Three-Compartment Muscle Fatigue Model: Invariance, Closed-Form Rest Allowances, and a Certified Shield for Human–Robot Collaboration"**, *Mathematics* (MDPI), 2026.

Author: Claudio Urrea ([ORCID 0000-0001-7197-8928](https://orcid.org/0000-0001-7197-8928)), Electrical Engineering Department, University of Santiago of Chile.

Archived release: https://doi.org/10.6084/m9.figshare.33960379 · License: MIT

## What is here

| Path | Content |
|---|---|
| `src/fatiga.py` | 3CC-r fatigue model (Xia & Frey-Law 2008; Looft et al. 2018), simplex-preserving integrator, closed-form equilibrium and sustainable load. `python fatiga.py verificar` reproduces the verification table |
| `src/verificar_periodico.py` | Closed-form periodic peak vs. full model and minimum rest allowance checks |
| `src/figuras.py`, `src/estilo_figuras.py` | Figures of the periodic regime, rest-allowance curves and Euler bound |
| `src/escudo_experimento.py` | Shield experiment: 3 allocators × with/without shield × 2 demand levels × 30 eight-hour shifts |
| `src/figura_traces_v2.py` | Shield-trace figure: 2-min min/max envelopes plus the per-shift maximum over all seeds |
| `src/revision_experimentos.py` | Experiments added in revision round 1 (see below) |
| `results/resultados_escudo.json` | Raw results of the shield experiment |
| `results/resultados_revision.json` | Raw results of the revision experiments |
| `robodk/crear_escena_hrc.py`, `robodk/visibilidad.py` | Programmatic construction of the three-operator, two-robot benchmark cell in RoboDK, with automatic reachability and collision checks, and reproducible camera views |
| `figures/` | All figures of the article (PDF and PNG) |

## Parameters (shoulder)

F = 0.01820 s⁻¹, R = 0.00168 s⁻¹ (Frey-Law et al. 2012, Table 1); r = 15 (general value of Looft et al. 2018; no shoulder data were available there); L_D = L_R = 10 (Xia & Frey-Law 2008).

## Reproducing the results

All scripts write their output to the current working directory and take no
absolute paths; run them from `src/`.

```bash
pip install -r requirements.txt
cd src
python fatiga.py verificar          # analytical results, seed 42
python verificar_periodico.py       # periodic peak and rest allowance table
python figuras.py                   # Figures 1-3
python escudo_experimento.py        # shield table and Figure 6 (seeds 1000-1029)
python figura_traces_v2.py          # Figure 5
python revision_experimentos.py     # Tables 1, 3, 5, 6, 7 of the article
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 and Numba 0.68;
with these versions the scripts reproduce the files in `results/` exactly.


`revision_experimentos.py` runs seven experiments:

| Block | Reproduces |
|---|---|
| `E0` | Explicit bound on the reduced-model error ε and its comparison with the measured error |
| `E1` | Tightness of the step-size bound in each of its three branches |
| `E2` | Tie-break rules among several admissible operators, cross-trained cell |
| `E3` | Sensitivity of the rest allowance and of the shield to the rest multiplier r |
| `E4` | Robustness to inter-individual variation in (F, R), and the two remedies |
| `E5` | Threshold sweep: safety and throughput against θ |
| `E6` | Paired statistics on common random numbers |

Seeds: 42 (verification), 1000–1029 (shield experiment), 2000–2029 and 3000–3029
(cross-trained cell), 7000–7119 and 9000–9059 (parameter robustness).

## Figures

All figures use Palatino Linotype, or its free clone TeX Gyre Pagella, which is
the Palatino family used by the MDPI template, through `src/estilo_figuras.py`.
Negative numbers are typeset with U+2212 (MINUS SIGN) rather than U+002D;
`estilo_figuras.verificar_minus()` checks that the chosen font provides the
glyph, and `estilo_figuras.auditar_pdf()` audits a generated PDF for hyphens
standing in front of a digit.

The RoboDK cell (optional) requires RoboDK and `pip install robodk`; open an
empty station and run `python crear_escena_hrc.py` from `robodk/`, then
`python visibilidad.py mixto`. The Doosan H2017 and A0509 robot files must be
downloaded from the RoboDK online library.

## Versions

| Release | Content |
|---|---|
| v1.0.0 | Code, results and figures of the submitted article |
| v1.1.0 | Experiments and figures added in revision round 1 |
| v1.1.1 | Final layout of Figure 5 for the published article (reference-level labels and legend moved clear of the data); no change to code results or data |

## Citation

See `CITATION.cff`. Please cite the article and the archived release.
