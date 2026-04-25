
// Multilevel Biased Agent Model: Centered Parameterization (CP)
data {
  int<lower=1> N_total;                 // Total number of observations
  int<lower=1> N_subjects;              // Total number of agents
  array[N_total] int<lower=1, upper=N_subjects> subj_id; // Agent ID index
  array[N_total] int<lower=0, upper=1> y;     // Observed choices
  real prior_mu_theta_mu;
  real<lower=0> prior_mu_theta_sigma;
  real<lower=0> prior_sigma_theta_lambda;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  // Population-level hyperparameters
  real mu_theta;                        // Population mean bias (logit scale)
  real<lower=0> sigma_theta;            // Population SD of bias (logit scale)

  // Individual-level parameters (CENTERED)
  vector[N_subjects] theta_logit;       // Agent-specific biases 
}

model {
  // 1. Population-level hyperpriors
  target += normal_lpdf(mu_theta | prior_mu_theta_mu, prior_mu_theta_sigma);
  target += exponential_lpdf(sigma_theta | prior_sigma_theta_lambda);

  // 2. Hierarchical prior (CENTERED)
  target += normal_lpdf(theta_logit | mu_theta, sigma_theta);

  // 3. Likelihood
  target += bernoulli_logit_lpmf(y | theta_logit[subj_id]); 
}

generated quantities {
  // Transform hyperparameters back to the probability (scale)
  real<lower=0, upper=1> mu_theta_prob = inv_logit(mu_theta);
  vector<lower=0, upper=1>[N_subjects] theta_prob = inv_logit(theta_logit);

  // Prior samples — cheap scalar draws; declared at top level so they appear in draws output
  real mu_theta_prior    = normal_rng(prior_mu_theta_mu, prior_mu_theta_sigma);
  real sigma_theta_prior = exponential_rng(prior_sigma_theta_lambda);

  // Initialize arrays for conditional diagnostics
  vector[N_total] log_lik;
  array[N_total] int y_post_rep;
  array[N_total] int y_prior_rep;

  // Accumulator for joint prior log-density
  real lprior;
  lprior = normal_lpdf(mu_theta | prior_mu_theta_mu, prior_mu_theta_sigma) +
           exponential_lpdf(sigma_theta | prior_sigma_theta_lambda) +
           normal_lpdf(theta_logit | mu_theta, sigma_theta);

  if (run_diagnostics) {
    for (i in 1:N_total) {
      log_lik[i] = bernoulli_logit_lpmf(y[i] | theta_logit[subj_id[i]]);
      y_post_rep[i] = bernoulli_logit_rng(theta_logit[subj_id[i]]);
      real theta_logit_prior_i = normal_rng(mu_theta_prior, sigma_theta_prior);
      y_prior_rep[i] = bernoulli_logit_rng(theta_logit_prior_i);
    }
  }
}

