
// Simple Bayesian Agent (SBA).
// No free parameters — evidence is counted at face value.
// Jeffreys prior pseudo-counts: alpha0 = beta0 = 0.5.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

model {
  // Vectorized likelihood with fixed weights = 1
  vector[N] alpha_post = 0.5 + to_vector(blue1) + to_vector(blue2);
  vector[N] beta_post  = 0.5 + (to_vector(total1) - to_vector(blue1))
                             + (to_vector(total2) - to_vector(blue2));
                             
  target += beta_binomial_lpmf(choice | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int posterior_pred;

  for (i in 1:N) {
    real alpha_post = 0.5 + blue1[i] + blue2[i];
    real beta_post  = 0.5 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}

