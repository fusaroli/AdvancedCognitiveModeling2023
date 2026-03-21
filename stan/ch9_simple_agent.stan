
// Simple Bayesian Agent: equal unit weights, no free parameters.
// alpha_prior = beta_prior = 1 fixed in transformed data.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

transformed data {
  // The uniform prior is fixed. This is not a free parameter.
  real alpha0 = 1.0;
  real beta0  = 1.0;
}

parameters {
  // No free parameters. The SBA is a deterministic prediction.
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
    // Prior predictive: uniform prior, no evidence
    prior_pred[i]     = beta_binomial_rng(1, alpha0, beta0);
  }
}

