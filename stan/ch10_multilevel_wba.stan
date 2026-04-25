
// Multilevel Weighted Bayesian Agent (reparameterised)
// Population distribution on (logit(rho), log(kappa)).
// Non-centred parameterisation with correlated random effects.
data {
  int<lower=1> N_total;                                // total observations
  int<lower=1> N_subjects;                             // number of agents
  array[N_total] int<lower=1, upper=N_subjects> subj_id;
  array[N_total] int<lower=0, upper=1>  y;
  array[N_total] int<lower=0>            blue1;
  array[N_total] int<lower=0>            total1;
  array[N_total] int<lower=0>            blue2;
  array[N_total] int<lower=0>            total2;
  
  real prior_mu_logit_rho_mu;
  real<lower=0> prior_mu_logit_rho_sigma;
  real prior_mu_log_kappa_mu;
  real<lower=0> prior_mu_log_kappa_sigma;
  real<lower=0> prior_sigma_lambda;
  real<lower=0> prior_Omega_shape;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  vector[2] mu;                              // population means: (logit_rho, log_kappa)
  vector<lower=0>[2] sigma;                  // population SDs
  cholesky_factor_corr[2] L_Omega;           // Cholesky factor of correlation matrix
  matrix[2, N_subjects] z;                   // standard normal deviates (NCP)
}

transformed parameters {
  // Reconstruct individual parameters on natural scale
  matrix[2, N_subjects] theta_raw = diag_pre_multiply(sigma, L_Omega) * z;
  
  vector<lower=0, upper=1>[N_subjects] rho;
  vector<lower=0>[N_subjects] kappa;
  vector<lower=0>[N_subjects] weight_direct;
  vector<lower=0>[N_subjects] weight_social;

  for (j in 1:N_subjects) {
    rho[j]           = inv_logit(mu[1] + theta_raw[1, j]);
    kappa[j]         = exp(mu[2] + theta_raw[2, j]);
    weight_direct[j] = rho[j] * kappa[j];
    weight_social[j] = (1.0 - rho[j]) * kappa[j];
  }
}

model {
  // Population priors
  target += normal_lpdf(mu[1] | prior_mu_logit_rho_mu, prior_mu_logit_rho_sigma);
  target += normal_lpdf(mu[2] | prior_mu_log_kappa_mu, prior_mu_log_kappa_sigma);
  target += exponential_lpdf(sigma | prior_sigma_lambda);
  target += lkj_corr_cholesky_lpdf(L_Omega | prior_Omega_shape);
  target += std_normal_lpdf(to_vector(z));
  
  // Likelihood (vectorised per trial)
  for (i in 1:N_total) {
    int j = subj_id[i];
    real alpha_post = 0.5
                    + weight_direct[j] * blue1[i]
                    + weight_social[j] * blue2[i];
    real beta_post  = 0.5
                    + weight_direct[j] * (total1[i] - blue1[i])
                    + weight_social[j] * (total2[i] - blue2[i]);
    target += beta_binomial_lpmf(y[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Population summaries on natural scale
  real pop_rho   = inv_logit(mu[1]);
  real pop_kappa = exp(mu[2]);
  matrix[2, 2] Omega = multiply_lower_tri_self_transpose(L_Omega);
  
  vector[N_total] log_lik;
  array[N_total] int y_rep;

  if (run_diagnostics) {
    for (i in 1:N_total) {
      int j = subj_id[i];
      real a = 0.5 + weight_direct[j] * blue1[i] + weight_social[j] * blue2[i];
      real b = 0.5 + weight_direct[j] * (total1[i] - blue1[i]) 
                   + weight_social[j] * (total2[i] - blue2[i]);
      log_lik[i] = beta_binomial_lpmf(y[i] | 1, a, b);
      y_rep[i]   = beta_binomial_rng(1, a, b);
    }
  }
}

