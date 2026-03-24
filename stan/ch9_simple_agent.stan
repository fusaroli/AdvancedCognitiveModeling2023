
// Simple Bayesian Agent — Jeffreys prior (alpha0 = beta0 = 0.5).
// No free parameters: the SBA is a deterministic prediction machine.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

transformed data {
  // Jeffreys prior for a binomial proportion: Beta(0.5, 0.5).
  // Parameterisation-invariant; concentrates slightly more mass near 0 and 1
  // than the uniform Beta(1, 1). With 8+ observations per trial the
  // difference is negligible, but the choice is principled.
  real alpha0 = 0.5;
  real beta0  = 0.5;
}

parameters {
  // No free parameters.
}

model {
  for (i in 1:N) {
    real alpha_post = alpha0 + blue1[i] + blue2[i];
    real beta_post  = beta0
                    + (total1[i] - blue1[i])
                    + (total2[i] - blue2[i]);
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;

  for (i in 1:N) {
    real alpha_post = alpha0 + blue1[i] + blue2[i];
    real beta_post  = beta0
                    + (total1[i] - blue1[i])
                    + (total2[i] - blue2[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);
    // Prior predictive: Jeffreys pseudo-counts only, no trial evidence
    prior_pred[i]     = beta_binomial_rng(1, alpha0, beta0);
  }
}

