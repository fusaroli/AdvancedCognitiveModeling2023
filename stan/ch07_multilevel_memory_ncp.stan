
// Hierarchical Memory Agent Model (Non-Centered Parameterization)
// alpha_j (baseline bias) and beta_j (memory sensitivity) are jointly
// drawn from a bivariate Normal. NCP decouples sampling geometry from
// the population parameters, preventing Neal's Funnel.

data {
  int<lower=1> N_total;                    // Total observations
  int<lower=1> N_subjects;                 // Number of agents
  array[N_total] int<lower=1, upper=N_subjects>  subj_id;   // Agent index
  array[N_total] int<lower=0, upper=1>  y;       // Observed choices
  vector<lower=0.01, upper=0.99>[N_total] opp_rate_prev;
  
  real prior_mu_alpha_mu;
  real<lower=0> prior_mu_alpha_sigma;
  real prior_mu_beta_mu;
  real<lower=0> prior_mu_beta_sigma;
  real<lower=0> prior_sigma_lambda;
  real<lower=0> prior_Omega_shape;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  // Population-level parameters
  real mu_alpha;              // Population mean baseline bias (logit scale)
  real mu_beta;               // Population mean memory sensitivity
  vector<lower=0>[2] sigma;   // Marginal SDs

  // Cholesky factor of the correlation matrix
  cholesky_factor_corr[2] L_Omega;

  // Individual-level parameters (NCP)
  matrix[2, N_subjects] z;
}

transformed parameters {
  matrix[N_subjects, 2] indiv_params;
  {
    matrix[2, N_subjects] deviations = diag_pre_multiply(sigma, L_Omega) * z;
    for (j in 1:N_subjects) {
      indiv_params[j, 1] = mu_alpha + deviations[1, j];
      indiv_params[j, 2] = mu_beta  + deviations[2, j];
    }
  }
}

model {
  // Hyperpriors
  target += normal_lpdf(mu_alpha | prior_mu_alpha_mu, prior_mu_alpha_sigma);
  target += normal_lpdf(mu_beta  | prior_mu_beta_mu,  prior_mu_beta_sigma);
  target += exponential_lpdf(sigma | prior_sigma_lambda);
  target += lkj_corr_cholesky_lpdf(L_Omega | prior_Omega_shape);

  // Individual-level prior (NCP)
  target += std_normal_lpdf(to_vector(z));

  // Likelihood
  vector[N_total] logit_p;
  for (i in 1:N_total) {
    logit_p[i] = indiv_params[subj_id[i], 1]
               + indiv_params[subj_id[i], 2] * logit(opp_rate_prev[i]);
  }
  target += bernoulli_logit_lpmf(y | logit_p);
}

generated quantities {
  matrix[2, 2] Omega = multiply_lower_tri_self_transpose(L_Omega);
  real rho = Omega[1, 2];

  vector[N_subjects] alpha    = indiv_params[, 1];
  vector[N_subjects] beta_mem = indiv_params[, 2];

  // Prior log-density
  real lprior = normal_lpdf(mu_alpha | prior_mu_alpha_mu, prior_mu_alpha_sigma)
              + normal_lpdf(mu_beta  | prior_mu_beta_mu,  prior_mu_beta_sigma)
              + exponential_lpdf(sigma | prior_sigma_lambda)
              + lkj_corr_cholesky_lpdf(L_Omega | prior_Omega_shape)
              + std_normal_lpdf(to_vector(z));

  // Predictive checks and pointwise log-likelihood
  vector[N_total] log_lik;
  array[N_total] int y_post_rep;
  array[N_total] int y_prior_rep;

  // Prior draws for prior-posterior update plots
  real mu_alpha_prior    = normal_rng(prior_mu_alpha_mu, prior_mu_alpha_sigma);
  real mu_beta_prior     = normal_rng(prior_mu_beta_mu, prior_mu_beta_sigma);
  real sigma_alpha_prior = exponential_rng(prior_sigma_lambda);
  real sigma_beta_prior  = exponential_rng(prior_sigma_lambda);
  // LKJ(2) marginal on rho for 2x2: 2*Beta(2,2) - 1
  real rho_prior         = 2.0 * beta_rng(2.0, 2.0) - 1.0;

  if (run_diagnostics) {
    matrix[2, 2] L_Omega_prior = lkj_corr_cholesky_rng(2, prior_Omega_shape);
    vector[2] sigma_prior = [sigma_alpha_prior, sigma_beta_prior]';
    matrix[2, N_subjects] z_prior;
    for (j in 1:N_subjects) { 
      z_prior[1,j] = normal_rng(0,1); 
      z_prior[2,j] = normal_rng(0,1); 
    }
    matrix[N_subjects, 2] indiv_prior;
    {
      matrix[2, N_subjects] dev_prior = diag_pre_multiply(sigma_prior, L_Omega_prior) * z_prior;
      for (j in 1:N_subjects) {
        indiv_prior[j, 1] = mu_alpha_prior + dev_prior[1, j];
        indiv_prior[j, 2] = mu_beta_prior  + dev_prior[2, j];
      }
    }

    for (i in 1:N_total) {
      int j = subj_id[i];
      real lp_post  = indiv_params[j,1] + indiv_params[j,2] * logit(opp_rate_prev[i]);
      real lp_prior = indiv_prior[j,1]  + indiv_prior[j,2]  * logit(opp_rate_prev[i]);

      log_lik[i]     = bernoulli_logit_lpmf(y[i] | lp_post);
      y_post_rep[i]  = bernoulli_logit_rng(lp_post);
      y_prior_rep[i] = bernoulli_logit_rng(lp_prior);
    }
  }
}

