
// Proportional Bayesian Agent (PBA).
// p in [0,1] allocates the unit evidence budget between direct and social.
// p = 0.5 approximates balanced weighting; p -> 1 ignores social; p -> 0 ignores direct.
data {
  int<lower=1> N_total;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total] int<lower=0> blue1;
  array[N_total] int<lower=0> total1;
  array[N_total] int<lower=0> blue2;
  array[N_total] int<lower=0> total2;
  
  real prior_p_alpha;
  real prior_p_beta;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real<lower=0, upper=1> p;  // Allocation to direct evidence
}

model {
  // prior
  target += beta_lpdf(p | prior_p_alpha, prior_p_beta);

  // Vectorized likelihood
  vector[N_total] alpha_post = 0.5 + p * to_vector(blue1) + (1.0 - p) * to_vector(blue2);
  vector[N_total] beta_post  = 0.5 + p * (to_vector(total1) - to_vector(blue1))
                                   + (1.0 - p) * (to_vector(total2) - to_vector(blue2));
                             
  target += beta_binomial_lpmf(y | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N_total] log_lik;
  array[N_total] int y_rep;
  real lprior = beta_lpdf(p | prior_p_alpha, prior_p_beta);

  if (run_diagnostics) {
    for (i in 1:N_total) {
      real alpha_post = 0.5 + p * blue1[i] + (1.0 - p) * blue2[i];
      real beta_post  = 0.5 + p * (total1[i] - blue1[i]) 
                           + (1.0 - p) * (total2[i] - blue2[i]);

      log_lik[i] = beta_binomial_lpmf(y[i] | 1, alpha_post, beta_post);
      y_rep[i]   = beta_binomial_rng(1, alpha_post, beta_post);
    }
  }
}

