
// Weighted Bayesian Agent (raw parameterisation).
// weight_direct, weight_social > 0: independent weights on direct and social evidence.
// Jeffreys prior pseudo-counts (0.5) consistent with SBA.
data {
  int<lower=1> N_total;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total] int<lower=0> blue1;
  array[N_total] int<lower=0> total1;
  array[N_total] int<lower=0> blue2;
  array[N_total] int<lower=0> total2;
  
  real prior_wd_mu;
  real<lower=0> prior_wd_sigma;
  real prior_ws_mu;
  real<lower=0> prior_ws_sigma;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real<lower=0> weight_direct;  // w_d
  real<lower=0> weight_social;  // w_s
}

model {
  // Priors: lognormal centered on 1 (evidence taken at face value)
  target += lognormal_lpdf(weight_direct | prior_wd_mu, prior_wd_sigma);
  target += lognormal_lpdf(weight_social | prior_ws_mu, prior_ws_sigma);

  // Vectorized likelihood
  vector[N_total] alpha_post = 0.5 + weight_direct * to_vector(blue1)
                                   + weight_social * to_vector(blue2);
  vector[N_total] beta_post  = 0.5 + weight_direct * (to_vector(total1) - to_vector(blue1))
                                   + weight_social * (to_vector(total2) - to_vector(blue2));
                             
  target += beta_binomial_lpmf(y | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N_total] log_lik;
  array[N_total] int y_rep;

  if (run_diagnostics) {
    for (i in 1:N_total) {
      real alpha_post = 0.5 + weight_direct * blue1[i] + weight_social * blue2[i];
      real beta_post  = 0.5 + weight_direct * (total1[i] - blue1[i]) 
                           + weight_social * (total2[i] - blue2[i]);

      log_lik[i] = beta_binomial_lpmf(y[i] | 1, alpha_post, beta_post);
      y_rep[i]   = beta_binomial_rng(1, alpha_post, beta_post);
    }
  }
}

