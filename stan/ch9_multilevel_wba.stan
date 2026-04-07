
// Multilevel Weighted Bayesian Agent (reparameterised)
// Population distribution on (logit(rho), log(kappa)).
// Non-centred parameterisation with correlated random effects.
data {
  int<lower=1> N;                                // total observations
  int<lower=1> J;                                // number of agents
  array[N] int<lower=1, upper=J> agent_id;
  array[N] int<lower=0, upper=1>  choice;
  array[N] int<lower=0>            blue1;
  array[N] int<lower=0>            total1;
  array[N] int<lower=0>            blue2;
  array[N] int<lower=0>            total2;
}

parameters {
  vector[2] mu;                              // population means: (logit_rho, log_kappa)
  vector<lower=0>[2] sigma;                  // population SDs
  cholesky_factor_corr[2] L_Omega;           // Cholesky factor of correlation matrix
  matrix[2, J] z;                            // standard normal deviates (NCP)
}

transformed parameters {
  // Reconstruct individual parameters on natural scale
  matrix[2, J] theta_raw = diag_pre_multiply(sigma, L_Omega) * z;
  
  vector<lower=0, upper=1>[J] rho;
  vector<lower=0>[J] kappa;
  vector<lower=0>[J] weight_direct;
  vector<lower=0>[J] weight_social;

  for (j in 1:J) {
    rho[j]           = inv_logit(mu[1] + theta_raw[1, j]);
    kappa[j]         = exp(mu[2] + theta_raw[2, j]);
    weight_direct[j] = rho[j] * kappa[j];
    weight_social[j] = (1.0 - rho[j]) * kappa[j];
  }
}

model {
  // Population priors
  target += normal_lpdf(mu[1] | 0, 1.5);       // logit_rho: weakly informative
  target += normal_lpdf(mu[2] | log(2), 0.5);   // log_kappa: centered on SBA-like scaling
  target += exponential_lpdf(sigma | 2);         // moderate shrinkage on SDs
  target += lkj_corr_cholesky_lpdf(L_Omega | 2); // weakly regularise correlation
  target += std_normal_lpdf(to_vector(z));        // NCP deviates
  
  // Likelihood (vectorised per trial)
  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 0.5
                    + weight_direct[j] * blue1[i]
                    + weight_social[j] * blue2[i];
    real beta_post  = 0.5
                    + weight_direct[j] * (total1[i] - blue1[i])
                    + weight_social[j] * (total2[i] - blue2[i]);
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Population summaries on natural scale
  real pop_rho   = inv_logit(mu[1]);
  real pop_kappa = exp(mu[2]);
  matrix[2, 2] Omega = multiply_lower_tri_self_transpose(L_Omega);
  
  vector[N] log_lik;
  array[N] int pred_choice;

  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 0.5
                    + weight_direct[j] * blue1[i]
                    + weight_social[j] * blue2[i];
    real beta_post  = 0.5
                    + weight_direct[j] * (total1[i] - blue1[i])
                    + weight_social[j] * (total2[i] - blue2[i]);
    log_lik[i]     = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}

