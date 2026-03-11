
// Multilevel Biased Agent Model (Non-Centered Parameterization)
// Estimates individual biases drawn from a population distribution.

data {
  int<lower=1> N;                    // Total number of observations
  int<lower=1> J;                    // Number of agents
  array[N] int<lower=1, upper=J> agent; // Agent ID for each observation (must be 1...J)
  array[N] int<lower=0, upper=1> h;    // Observed choices (0 or 1)
}

parameters {
  // Population-level parameters
  real mu_theta;                     // Population mean bias (logit scale)
  real<lower=0> sigma_theta;         // Population SD of bias (logit scale)

  // Individual-level parameters (standardized deviations, z-scores of the agents)
  vector[J] z_theta;                 // Non-centered individual effects (standard normal scale)
}

transformed parameters {
  // Deterministic reconstruction of the cognitive parameters
  // This shifts and scales the z-scores back to the target logit space
  vector[J] theta_logit = mu_theta + z_theta * sigma_theta;
}

model {
  // Priors for population-level parameters
  target += normal_lpdf(mu_theta | 0, 1.5);
  target += exponential_lpdf(sigma_theta | 1);
  
  /// Individual level prior (NON-CENTERED)
  // Notice there are no population level parameters here. The geometry is decoupled.
  target += std_normal_lpdf(z_theta);

  // Likelihood
  /// The likelihood is evaluated on the deterministically transformed parameters
  /// not the raw z-scores.
  target += bernoulli_logit_lpmf(h | theta_logit[agent]);
}

generated quantities {
  // Transform hyperparameters back to the probability (outcome) scale
  real<lower=0, upper=1> mu_theta_prob = inv_logit(mu_theta);
  vector<lower=0, upper=1>[J] theta_prob = inv_logit(theta_logit);
  
  // Initialize containers for trial-level metrics
  vector[N] log_lik;
  array[N] int h_post_rep;
  array[N] int h_prior_rep;

  // --- Prior Log-Density  ---
  real lprior;
  // Notice we score z_theta with the standard normal distribution, exactly as in the model block
  lprior = normal_lpdf(mu_theta | 0, 1.5) + 
           exponential_lpdf(sigma_theta | 1) + 
           std_normal_lpdf(z_theta);

  // --- Prior Predictive Checks: Generative Baseline ---
  // 1. Draw population hyperparameters from the EXACT priors used in the model block
  real mu_theta_prior = normal_rng(0, 1.5);
  real<lower=0> sigma_theta_prior = exponential_rng(1);
  
  // 2. Draw individual agent parameters from the non-centered prior
  vector[J] theta_logit_prior;
  for (j in 1:J) {
    real z_theta_prior = normal_rng(0, 1); 
    theta_logit_prior[j] = mu_theta_prior + z_theta_prior * sigma_theta_prior;
  }

  // --- Trial-Level Computations ---
  // We consolidate the log-likelihood and predictive checks into a single loop
  for (i in 1:N) {
    // 1. Pointwise log-likelihood (for PSIS-LOO model comparison later)
    log_lik[i] = bernoulli_logit_lpmf(h[i] | theta_logit[agent[i]]);
    
    // 2. Posterior Predictive Check: Generate choices using the fitted posterior
    h_post_rep[i] = bernoulli_logit_rng(theta_logit[agent[i]]);
    
    // 3. Prior Predictive Check: Generate choices using the unconditioned prior
    h_prior_rep[i] = bernoulli_logit_rng(theta_logit_prior[agent[i]]);
  }
}
