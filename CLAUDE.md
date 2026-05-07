# Bayesian Cognitive Modeling Specialist — System Specification

This project follows the strict Bayesian modeling and validation standards defined in the *Advanced Cognitive Modeling* book. All cognitive modeling tasks must adhere to the **six-phase validation battery**.

## PHASE 0: CONCEPTUAL ARCHITECTURE & INTERROGATION
Before implementation, resolve:
1. **The Cognitive-to-Statistical Mapping:** Verbal theory as an algorithm.
2. **Interpretability:** Psychological mechanism per parameter.
3. **Experimental Structure:** Nesting and path-dependency.
4. **Data Reality:** Lapses and missing data handling.
5. **Identifiability:** Parameter confounding (e.g., bias vs. memory).

## THE SIX-PHASE VALIDATION BATTERY
Every model must pass these six phases before fitting real data:
1. **Mathematical Formulation:** Verify Stan code matches intent.
2. **Prior Predictive (PPC):** Check generative plausibility.
3. **Posterior Predictive:** Check fit to key data patterns.
4. **Prior-Posterior Update:** Quantify information gain.
5. **Sensitivity Analysis:** Robustness to prior choices.
6. **Recovery + SBC:** Calibration and unbiased estimation.

## TECHNICAL STANDARDS

### 1. Stan Implementation (`cmdstanr`)
- **R-First Pre-computation:** Compute data-only quantities in R.
- **Vectorization:** Minimize autodiff nodes.
- **Block Discipline:** Use `transformed parameters` only for gradients; `generated quantities` for validation.
- **Initialization (Pathfinder):** Use `mod$pathfinder()` for warm-start; provide jittered prior-median fallback.

### 2. Parameterization & Priors
- **Probabilities [0,1]:** Logit transform (`theta_logit ~ normal(0, 1.5)`).
- **Positive Scales:** Log transform or `exponential(1)` priors.
- **Hierarchical Models:** Use **Non-Centered Parameterization (NCP)** by default.
- **Correlations:** Use Cholesky factors with `lkj_corr_cholesky(2)`.
- **Numerical Stability:** Use `log_sum_exp()`, `log_mix()`, and `bernoulli_logit_lpmf()`.

### 3. Model Comparison & Diagnostics
- **MCMC Diagnostics:** $\hat{R} < 1.01$, Bulk/Tail ESS $> 400$, 0 divergences.
- **PSIS-LOO:** Identify influential observations via Pareto-$\hat{k} > 0.7$.
- **SBC:** Use the `SBC` package for calibration.

## R PACKAGE STACK
Use `pacman::p_load(tidyverse, posterior, cmdstanr, tidybayes, patchwork, bayesplot, loo, priorsense, SBC, future, furrr)`.

---
**When asked to implement a model, state the mathematical specification (LaTeX), provide the R simulator, and generate the raw Stan code following these standards.**
