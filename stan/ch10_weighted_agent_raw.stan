
// Weighted Bayesian Agent (raw parameterisation).
// w_d, w_s > 0: independent weights on direct and social evidence.
// Jeffreys prior pseudo-counts (0.5) consistent with SBA.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

parameters {
  real<lower=0> weight_direct;  // w_d
  real<lower=0> weight_social;  // w_s
}

model {
  // Priors: lognormal centered on 1 (evidence taken at face value)
  target += lognormal_lpdf(weight_direct | 0, 0.5);
  target += lognormal_lpdf(weight_social | 0, 0.5);

  // Vectorized likelihood
  vector[N] alpha_post = 0.5 + weight_direct * to_vector(blue1)
                             + weight_social * to_vector(blue2);
  vector[N] beta_post  = 0.5 + weight_direct * (to_vector(total1) - to_vector(blue1))
                             + weight_social * (to_vector(total2) - to_vector(blue2));
                             
  target += beta_binomial_lpmf(choice | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int posterior_pred;

  for (i in 1:N) {
    real alpha_post = 0.5 + weight_direct * blue1[i] + weight_social * blue2[i];
    real beta_post  = 0.5 + weight_direct * (total1[i] - blue1[i]) 
                         + weight_social * (total2[i] - blue2[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}

