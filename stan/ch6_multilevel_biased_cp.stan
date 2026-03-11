
// Multilevel Biased Agent Model: Centered Parameterization (CP)
data {
  int<lower=1> N;                       // Total number of observations across all agents
  int<lower=1> J;                       // Total number of agents
  array[N] int<lower=1, upper=J> agent; // Agent ID index for each trial
  array[N] int<lower=0, upper=1> h;     // Observed binary choices (1 = Right, 0 = Left)
}

parameters {
  // Population-level hyperparameters
  real mu_theta;                        // Population mean bias (logit scale)
  real<lower=0> sigma_theta;            // Population SD of bias (logit scale)

  // Individual-level parameters (CENTERED)
  vector[J] theta_logit;                // Agent-specific biases 
}

model {
  // 1. Population-level hyperpriors
  target += normal_lpdf(mu_theta | 0, 1.5);
  target += exponential_lpdf(sigma_theta | 1);

  // 2. Hierarchical prior (CENTERED)
  // The individual parameters are directly parameterized by the hyperparameters.
  // This explicitly creates the funnel geometry.
  target += normal_lpdf(theta_logit | mu_theta, sigma_theta);

  // 3. Likelihood
  // Highly vectorized evaluation mapping each trial to its specific agent
  target += bernoulli_logit_lpmf(h | theta_logit[agent]); 
}

generated quantities {
  // Transform hyperparameters back to the probability (outcome) scale for interpretation
  real<lower=0, upper=1> mu_theta_prob = inv_logit(mu_theta);
  vector<lower=0, upper=1>[J] theta_prob = inv_logit(theta_logit);

  // Initialize arrays for predictive checks, LOO-CV, and Prior Sensitivity
  vector[N] log_lik;
  array[N] int h_post_rep;
  array[N] int h_prior_rep;
  
  // Accumulator for joint prior log-density (required by 'priorsense')
  real lprior; 
  lprior = normal_lpdf(mu_theta | 0, 1.5) + 
           exponential_lpdf(sigma_theta | 1) + 
           normal_lpdf(theta_logit | mu_theta, sigma_theta);

  // Draw from the priors to establish the generative baseline
  real mu_theta_prior = normal_rng(0, 1.5);
  real sigma_theta_prior = exponential_rng(1);
  
  // We use a loop for trial-level generation to ensure array indexing stability
  for (i in 1:N) {
    // Pointwise log-likelihood for out-of-sample predictive performance (PSIS-LOO)
    log_lik[i] = bernoulli_logit_lpmf(h[i] | theta_logit[agent[i]]);
    
    // Posterior Predictive Check (PPC): Data generated from the fitted posterior
    h_post_rep[i] = bernoulli_logit_rng(theta_logit[agent[i]]);
    
    // Prior Predictive Check: Data generated entirely from the unconditioned priors
    // We sample a prior theta for the specific agent on the fly
    real theta_logit_prior_i = normal_rng(mu_theta_prior, sigma_theta_prior);
    h_prior_rep[i] = bernoulli_logit_rng(theta_logit_prior_i);
  }
}

