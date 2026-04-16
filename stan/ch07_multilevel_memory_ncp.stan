
// Hierarchical Memory Agent Model (Non-Centered Parameterization)
// -------------------------------------------------------------------
// alpha_j (baseline bias) and beta_j (memory sensitivity) are jointly
// drawn from a bivariate Normal. NCP decouples sampling geometry from
// the population parameters, preventing Neal's Funnel.

data {
  int<lower=1> N;                          // Total observations across all agents
  int<lower=1> J;                          // Number of agents
  array[N] int<lower=1, upper=J>  agent;   // Agent index per trial
  array[N] int<lower=0, upper=1>  h;       // Observed binary choices
  // Pre-computed running average of opponent choices, clipped to [0.01, 0.99].
  vector<lower=0.01, upper=0.99>[N] opp_rate_prev;
}

parameters {
  // Population-level parameters
  real mu_alpha;              // Population mean baseline bias (logit scale)
  real mu_beta;               // Population mean memory sensitivity
  vector<lower=0>[2] sigma;   // Marginal SDs: sigma[1] = sigma_alpha, sigma[2] = sigma_beta

  // Cholesky factor of the 2x2 correlation matrix. Prior: Omega ~ LKJ(2).
  cholesky_factor_corr[2] L_Omega;

  // Individual-level parameters (NCP): 2 x J matrix of standard normals.
  // Row 1 = alpha offsets, row 2 = beta offsets.
  matrix[2, J] z;
}

transformed parameters {
  // Reconstruct cognitive parameters from z-space via the Cholesky transform.
  // diag_pre_multiply(sigma, L_Omega) = diag(sigma) * L_Omega  (2x2 matrix)
  // Multiplying by z (2xJ) and transposing gives J x 2.
  matrix[J, 2] indiv_params;
  {
    matrix[2, J] deviations = diag_pre_multiply(sigma, L_Omega) * z;
    for (j in 1:J) {
      indiv_params[j, 1] = mu_alpha + deviations[1, j];
      indiv_params[j, 2] = mu_beta  + deviations[2, j];
    }
  }
}

model {
  // Hyperpriors
  target += normal_lpdf(mu_alpha | 0, 1);
  target += normal_lpdf(mu_beta  | 0, 1);
  target += exponential_lpdf(sigma | 1);
  target += lkj_corr_cholesky_lpdf(L_Omega | 2);

  // Individual-level prior (NCP): geometry fully decoupled from population params
  target += std_normal_lpdf(to_vector(z));

  // Likelihood
  vector[N] logit_p;
  for (i in 1:N) {
    logit_p[i] = indiv_params[agent[i], 1]
               + indiv_params[agent[i], 2] * logit(opp_rate_prev[i]);
  }
  target += bernoulli_logit_lpmf(h | logit_p);
}

generated quantities {
  // Recover the full correlation matrix and the scalar rho
  matrix[2, 2] Omega = multiply_lower_tri_self_transpose(L_Omega);
  real rho = Omega[1, 2];

  // Individual parameters on natural scales
  vector[J] alpha    = indiv_params[, 1];
  vector[J] beta_mem = indiv_params[, 2];

  // Prior log-density (required by priorsense)
  real lprior = normal_lpdf(mu_alpha | 0, 1)
              + normal_lpdf(mu_beta  | 0, 1)
              + exponential_lpdf(sigma | 1)
              + lkj_corr_cholesky_lpdf(L_Omega | 2)
              + std_normal_lpdf(to_vector(z));

  // Prior draws for prior-posterior update plots
  real mu_alpha_prior    = normal_rng(0, 1);
  real mu_beta_prior     = normal_rng(0, 1);
  real sigma_alpha_prior = exponential_rng(1);
  real sigma_beta_prior  = exponential_rng(1);
  // LKJ(2) marginal on rho: 2*Beta(2,2) - 1
  real rho_prior         = 2.0 * beta_rng(2.0, 2.0) - 1.0;

  // Predictive checks and pointwise log-likelihood
  vector[N] log_lik;
  array[N] int h_post_rep;
  array[N] int h_prior_rep;

  // Build a prior-drawn population for h_prior_rep
  cholesky_factor_corr[2] L_Omega_prior = lkj_corr_cholesky_rng(2, 2.0);
  vector[2] sigma_prior = [sigma_alpha_prior, sigma_beta_prior]';
  matrix[2, J] z_prior;
  for (j in 1:J) { z_prior[1,j] = normal_rng(0,1); z_prior[2,j] = normal_rng(0,1); }
  matrix[J, 2] indiv_prior;
  {
    matrix[2, J] dev_prior = diag_pre_multiply(sigma_prior, L_Omega_prior) * z_prior;
    for (j in 1:J) {
      indiv_prior[j, 1] = mu_alpha_prior + dev_prior[1, j];
      indiv_prior[j, 2] = mu_beta_prior  + dev_prior[2, j];
    }
  }

  for (i in 1:N) {
    int j = agent[i];
    real lp_post  = indiv_params[j,1] + indiv_params[j,2] * logit(opp_rate_prev[i]);
    real lp_prior = indiv_prior[j,1]  + indiv_prior[j,2]  * logit(opp_rate_prev[i]);

    log_lik[i]     = bernoulli_logit_lpmf(h[i] | lp_post);
    h_post_rep[i]  = bernoulli_logit_rng(lp_post);
    h_prior_rep[i] = bernoulli_logit_rng(lp_prior);
  }
}

