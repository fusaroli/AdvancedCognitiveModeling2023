
data {
  int<lower=1> N_trials;               // Total number of trials
  array[N_trials] int<lower=0, upper=1> y; // Observed choices
  real prior_theta_mu;                 // Mean of the prior for theta_logit
  real<lower=0> prior_theta_sigma;     // SD of the prior for theta_logit
  int<lower=0, upper=1> run_diagnostics; // Flag: 1 to compute PPC/log_lik, 0 to skip
}
parameters {
  real theta_logit; // Represents the log-odds of choosing option 1 (right)
}
model {
  target += normal_lpdf(theta_logit | prior_theta_mu, prior_theta_sigma); // prior
  target += bernoulli_logit_lpmf(y | theta_logit); // likelihood
}

generated quantities {
  // Create the prior for theta
  real theta_logit_prior = normal_rng(prior_theta_mu, prior_theta_sigma); 
  // --- Parameter Transformation ---
  real theta_prior = inv_logit(theta_logit_prior); 
  real<lower=0, upper=1> theta = inv_logit(theta_logit);

  // --- Conditional Diagnostics ---
  // These blocks are expensive to run and store during SBC.
  // We wrap them in a conditional flag.
  array[N_trials] int y_prior_rep;
  int<lower=0, upper=N_trials> prior_rep_sum;
  array[N_trials] int y_post_rep;
  int<lower=0, upper=N_trials> post_rep_sum;
  vector[N_trials] log_lik;
  real lprior;

  if (run_diagnostics) {
    // --- Prior Predictive Check Simulation ---
    y_prior_rep = bernoulli_logit_rng(rep_vector(theta_logit_prior, N_trials));
    prior_rep_sum = sum(y_prior_rep);

    // --- Posterior Predictive Check Simulation ---
    y_post_rep = bernoulli_logit_rng(rep_vector(theta_logit, N_trials));
    post_rep_sum = sum(y_post_rep); 
    
    // --- Log-likelihood ---
    for (i in 1:N_trials) log_lik[i] = bernoulli_logit_lpmf(y[i] | theta_logit);
  }
  // joint log prior (useful for sensitivity even if run_diagnostics=0)
  lprior = normal_lpdf(theta_logit | prior_theta_mu, prior_theta_sigma);
}

