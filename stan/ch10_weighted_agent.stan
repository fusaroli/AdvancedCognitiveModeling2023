
// Weighted Bayesian Agent (reparameterised).
// rho in (0,1): relative weight of direct vs social evidence.
// kappa > 0: total evidence scaling (w_d + w_s).
data {
  int<lower=1> N_total;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total] int<lower=0> blue1;
  array[N_total] int<lower=0> total1;
  array[N_total] int<lower=0> blue2;
  array[N_total] int<lower=0> total2;
  
  real prior_rho_alpha;
  real prior_rho_beta;
  real prior_kappa_mu;
  real<lower=0> prior_kappa_sigma;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real<lower=0, upper=1> rho;    // relative weight: w_d / (w_d + w_s)
  real<lower=0>          kappa;  // total weight: w_d + w_s
}

transformed parameters {
  real<lower=0> weight_direct = rho * kappa;
  real<lower=0> weight_social = (1.0 - rho) * kappa;
}

model {
  // rho: weakly centred on equal weighting
  target += beta_lpdf(rho | prior_rho_alpha, prior_rho_beta);
  // kappa: lognormal centered on 2 (SBA equivalent)
  target += lognormal_lpdf(kappa | prior_kappa_mu, prior_kappa_sigma);

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
  real lprior = beta_lpdf(rho | prior_rho_alpha, prior_rho_beta) + 
                lognormal_lpdf(kappa | prior_kappa_mu, prior_kappa_sigma);

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

