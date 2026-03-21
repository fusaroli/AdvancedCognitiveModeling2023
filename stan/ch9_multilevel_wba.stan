
// Multilevel Weighted Bayesian Agent
// Non-centred parameterisation on log scale.
// Correlated random effects for (log_wd, log_ws) via LKJ prior.
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> agent_id;
  array[N] int<lower=0, upper=1>  choice;
  array[N] int<lower=0>            blue1;
  array[N] int<lower=0>            total1;
  array[N] int<lower=0>            blue2;
  array[N] int<lower=0>            total2;
}

parameters {
  // Population means on log scale
  vector[2] mu_log;                        // [mu_log_wd, mu_log_ws]

  // Population SDs (between-subject variability)
  vector<lower=0>[2] sigma_log;            // [sigma_log_wd, sigma_log_ws]

  // Cholesky factor of the 2x2 correlation matrix (Chapter 6 style)
  cholesky_factor_corr[2] L_Omega;

  // Standardised individual deviations (NCP): 2 x J matrix
  matrix[2, J] z;
}

transformed parameters {
  // Individual weights (positive via exp transform)
  vector<lower=0>[J] weight_direct;
  vector<lower=0>[J] weight_social;

  // Correlated NCP: theta_j = mu + diag(sigma) * L_Omega * z_j
  matrix[2, J] theta = diag_pre_multiply(sigma_log, L_Omega) * z;

  for (j in 1:J) {
    weight_direct[j] = exp(mu_log[1] + theta[1, j]);
    weight_social[j] = exp(mu_log[2] + theta[2, j]);
  }
}

model {
  // Population-level priors
  target += normal_lpdf(mu_log    | 0, 1);
  target += exponential_lpdf(sigma_log | 2);

  // LKJ prior on correlations (eta=2: weakly regularising, consistent with Ch.6)
  target += lkj_corr_cholesky_lpdf(L_Omega | 2);

  // Non-centred individual effects
  target += std_normal_lpdf(to_vector(z));

  // Likelihood
  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 1.0
                    + weight_direct[j] * blue1[i]
                    + weight_social[j] * blue2[i];
    real beta_post  = 1.0
                    + weight_direct[j] * (total1[i] - blue1[i])
                    + weight_social[j] * (total2[i] - blue2[i]);
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Population-level parameters on natural scale
  real pop_weight_direct = exp(mu_log[1]);
  real pop_weight_social = exp(mu_log[2]);

  // Recover the full correlation matrix for reporting
  matrix[2, 2] Omega = multiply_lower_tri_self_transpose(L_Omega);

  vector[N] log_lik;
  array[N] int pred_choice;

  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 1.0
                    + weight_direct[j] * blue1[i]
                    + weight_social[j] * blue2[i];
    real beta_post  = 1.0
                    + weight_direct[j] * (total1[i] - blue1[i])
                    + weight_social[j] * (total2[i] - blue2[i]);

    log_lik[i]     = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}

