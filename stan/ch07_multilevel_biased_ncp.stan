
// Multilevel Biased Agent Model (Non-Centered Parameterization)
// Estimates individual biases drawn from a population distribution.

data {
  int<lower=1> N_total;                 // Total number of observations
  int<lower=1> N_subjects;              // Number of agents
  array[N_total] int<lower=1, upper=N_subjects> subj_id; // Agent ID
  array[N_total] int<lower=0, upper=1> y;     // Observed choices
  real prior_mu_theta_mu;
  real<lower=0> prior_mu_theta_sigma;
  real<lower=0> prior_sigma_theta_lambda;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  // Population-level parameters
  real mu_theta;                     // Population mean bias (logit scale)
  real<lower=0> sigma_theta;         // Population SD of bias (logit scale)

  // Individual-level parameters (standardized deviations, z-scores of the agents)
  vector[N_subjects] z_theta;        // Non-centered individual effects
}

transformed parameters {
  // Deterministic reconstruction of the cognitive parameters
  vector[N_subjects] theta_logit = mu_theta + z_theta * sigma_theta;
}

model {
  // Priors for population-level parameters
  target += normal_lpdf(mu_theta | prior_mu_theta_mu, prior_mu_theta_sigma);
  target += exponential_lpdf(sigma_theta | prior_sigma_theta_lambda);
  
  /// Individual level prior (NON-CENTERED)
  target += std_normal_lpdf(z_theta);

  // Likelihood
  target += bernoulli_logit_lpmf(y | theta_logit[subj_id]);
}

generated quantities {
  // Transform hyperparameters back to the probability (scale)
  real<lower=0, upper=1> mu_theta_prob = inv_logit(mu_theta);
  vector<lower=0, upper=1>[N_subjects] theta_prob = inv_logit(theta_logit);
  
  // Initialize containers for trial-level metrics
  vector[N_total] log_lik;
  array[N_total] int y_post_rep;
  array[N_total] int y_prior_rep;

  // --- Prior Log-Density  ---
  real lprior;
  lprior = normal_lpdf(mu_theta | prior_mu_theta_mu, prior_mu_theta_sigma) + 
           exponential_lpdf(sigma_theta | prior_sigma_theta_lambda) + 
           std_normal_lpdf(z_theta);

  // --- Prior Predictive Checks: Generative Baseline ---
  real mu_theta_prior = normal_rng(prior_mu_theta_mu, prior_mu_theta_sigma);
  real sigma_theta_prior = exponential_rng(prior_sigma_theta_lambda);

  if (run_diagnostics) {
    // 2. Draw individual agent parameters from the non-centered prior
    vector[N_subjects] theta_logit_prior;
    for (j in 1:N_subjects) {
      real z_theta_prior = normal_rng(0, 1); 
      theta_logit_prior[j] = mu_theta_prior + z_theta_prior * sigma_theta_prior;
    }

    // --- Trial-Level Computations ---
    for (i in 1:N_total) {
      log_lik[i] = bernoulli_logit_lpmf(y[i] | theta_logit[subj_id[i]]);
      y_post_rep[i] = bernoulli_logit_rng(theta_logit[subj_id[i]]);
      y_prior_rep[i] = bernoulli_logit_rng(theta_logit_prior[subj_id[i]]);
    }
  }
}
