# BART-BO: Bayesian Additive Regression Trees for Constrained Combinatorial Bayesian Optimisation
 
> **Niyati Seth & Michael Fop** · November 2025
 
This repository contains the R code accompanying the paper *"Bayesian Additive Regression Trees for Constrained Combinatorial Bayesian Optimisation"*. The framework introduces a sparse BART-based surrogate model for black-box optimisation over binary combinatorial search spaces, with full support for explicit feasibility constraints.
 
---
 
## Overview
 
Standard Bayesian optimisation (BO) is designed for continuous, smooth domains. Extending it to **high-dimensional binary combinatorial spaces with black-box constraints** requires:
 
- A surrogate model that handles discrete inputs and nonlinear interactions natively
- Probabilistic modelling of both objective values and feasibility
- Acquisition optimisation that operates directly over `{0,1}^p`
This repository implements two methods:
 
| Method | Description |
|---|---|
| **BART-BO** | Unconstrained combinatorial BO using Dirichlet-BART surrogate + EI acquisition |
| **BART-CBO** | Constrained extension with joint modelling of feasibility probabilities and violation magnitudes |
 
Both methods use a **Dirichlet sparsity prior** (Linero, 2018) over BART splitting probabilities to focus model capacity on influential variables, implemented via [`dartMachine`](https://github.com/cran/bartMachine) — a modified version of the `bartMachine` R package. A **genetic algorithm** (R package `GA`) is used to optimise the acquisition function over the binary domain.
 
---
 
## Repository Structure
 
```
.
├── unconstrained/              # BART-BO: unconstrained binary optimisation
│   └── BQP/                   #   Binary Quadratic Programming (p = 100)
│                               #   Compares standard BART vs Dirichlet-BART (dartMachine)
│
├── constrained/                # BART-CBO: constrained combinatorial optimisation
│   ├── knapsack/               #   0/1 Knapsack Problem (p = 24 items)
│   ├── CBQP/                   #   Cardinality-Constrained BQP (p = 50, 100)
│   └── facility_location/      #   San Francisco facility location (LSCP & MCLP)
│
└── knapsack_testBART/          # Exploratory: standard BART (no Dirichlet prior) on knapsack
```
 
### Folder Details
 
#### `unconstrained/`
 
Runs the unconstrained Binary Quadratic Programming (BQP) benchmark with `p = 100` binary variables. The objective is `min x'Qx` where `Q` has structured exponential-decay interactions.
 
- Compares **standard BART** (`bartMachine`) against **Dirichlet-BART** (`dartMachine`)
- 250 BO iterations, 20 random initial points, 10 independent replications
- Benchmark comparison against a standalone genetic algorithm (5000 evaluations)
#### `constrained/knapsack/`
 
Applies **BART-CBO** to the classical 0/1 Knapsack Problem using a public benchmark instance (Kreher & Stinson, 2020) with `p = 24` items and known optimal value `13,549,094`.
 
- Evaluates penalty weights `λ ∈ {0.1, 1, 2, 5, 10}`
- 250 BO iterations with 10 random initial points, 10 replications per penalty setting
- Compares against GA benchmark (~3836 evaluations)
#### `constrained/CBQP/`
 
Extends the BQP benchmark with a cardinality constraint `∑ xⱼ ≤ s`, tested at:
- `p = 50`, `s = 40`
- `p = 100`, `s = 70`
Demonstrates BART-CBO's scalability and the effect of dimensionality on surrogate quality and feasibility rates.
 
#### `constrained/facility_location/`
 
Applies BART-CBO to two classical facility location formulations on the **San Francisco retail dataset** (205 census tract demand points, 16 candidate facility locations, 5 km road-network coverage radius):
 
- **LSCP** (Location Set Covering Problem): minimise number of facilities such that all demand points are covered. Tests initial sample sizes `N₀ ∈ {5, 10, 20}`. Global optimum: 8 facilities (verified with GLPK).
- **MCLP** (Maximal Covering Location Problem): maximise weighted demand coverage with a budget of `P = 4` facilities. Best BART-CBO solution covers 185/205 demand points (90.28% of population), matching the GLPK optimum.
#### `knapsack_testBART/`
 
Exploratory scripts testing the **standard BART surrogate** (without the Dirichlet sparsity prior) on the knapsack problem. Provides a direct baseline for assessing the benefit of the Dirichlet prior for constraint and feasibility modelling in `dartMachine`.
 
---
 
## Methods Summary
 
### BART-BO (Unconstrained)
 
1. Fit a Dirichlet-BART regression surrogate to normalised objective observations
2. Compute posterior predictive mean `μ̂(x)` and variance `σ̂²(x)`
3. Maximise Expected Improvement (EI) over `{0,1}^p` using a genetic algorithm
4. Evaluate true objective, update dataset, refit surrogate
### BART-CBO (Constrained)
 
Extends BART-BO with three surrogate components per constraint `cₖ`:
 
- **Objective surrogate**: Dirichlet-BART regression on normalised `f(x)` values
- **Feasibility surrogate**: Dirichlet-BART probit classifier on binary feasibility indicators `zₖ = I(cₖ(x) ≤ 0)`
- **Violation surrogate**: Dirichlet-BART regression on normalised violation magnitudes `vₖ = max(0, cₖ(x))`
The constrained acquisition function is:
 
```
fa(x) = EI(x) · p̂_feas(x) − λ · Σ v̂ₖ(x)
```
 
The penalty `λ` is updated adaptively: increased multiplicatively after consecutive infeasible proposals, decreased after a feasible solution is found.
 
---
 
## Dependencies
 
All code is written in **R**. The following packages are required:
 
```r
install.packages(c(
  "bartMachine",   # Base BART implementation
  "GA",            # Genetic algorithm for acquisition optimisation
  "Rglpk",         # GLPK solver (facility location reference solutions)
  "maxcovr",       # Facility location data and utilities
  "ggplot2",       # Plotting
  "dplyr"          # Data manipulation
))
```
 
`dartMachine` (Dirichlet-BART) is sourced from Linero (2018)'s modified `bartMachine` implementation. See the package documentation for installation instructions.
 
`bartMachine` requires a Java runtime. Ensure Java is installed and configured before use:
 
```r
options(java.parameters = "-Xmx4g")  # Allocate sufficient heap memory
library(bartMachine)
```
 
---
 
## Reproducing Results
 
Each subfolder contains self-contained R scripts. Run them independently:
 
```bash
# Unconstrained BQP (BART vs Dirichlet-BART comparison)
Rscript unconstrained/BQP/run_BQP.R
 
# Knapsack (constrained)
Rscript constrained/knapsack/run_knapsack.R
 
# Cardinality-constrained BQP
Rscript constrained/CBQP/run_CBQP.R
 
# Facility location (LSCP and MCLP)
Rscript constrained/facility_location/run_LSCP.R
Rscript constrained/facility_location/run_MCLP.R
 
# Standard BART baseline (knapsack)
Rscript knapsack_testBART/run_knapsack_BART.R
```
 
All experiments use fixed random seeds for reproducibility. Results are averaged over 10 independent replications.
 
---
 
## Key Results
 
| Problem | Method | Evaluations | Best Objective |
|---|---|---|---|
| BQP (p=100) | Dirichlet BART-BO | 270 | −45.77 ± 4.71 |
| BQP (p=100) | Standard BART-BO | 270 | −43.44 ± 6.66 |
| BQP (p=100) | GA | 5000 | −41.12 ± 4.68 |
| Knapsack (p=24) | BART-CBO (λ=5) | 260 | 13.14 × 10⁶ |
| Knapsack (p=24) | GA | ~3836 | 13.35 × 10⁶ |
| LSCP (p=16) | BART-CBO | 270 | 8 facilities (optimal) |
| MCLP (p=16) | BART-CBO | 260 | 185/205 demand points covered (optimal) |
 
BART-CBO consistently achieves near-GA solution quality using **an order of magnitude fewer objective function evaluations**.
 
---
 
## Citation
 
If you use this code in your research, please cite:
 
```bibtex
@article{seth2025bartbo,
  title   = {Bayesian Additive Regression Trees for Constrained Combinatorial Bayesian Optimisation},
  author  = {Seth, Niyati and Fop, Michael},
  year    = {2025},
  month   = {November}
}
```
