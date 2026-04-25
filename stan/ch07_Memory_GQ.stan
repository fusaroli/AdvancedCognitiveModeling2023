
// ch07_MemoryBernoulli_GQ.stan
// Social Memory Agent: Estimates bias and memory sensitivity.

data {
  int<lower=1> N_trials;               // Number of trials
  array[N_trials] int y;               // Agent's choices (0, 1)
  array[N_trials] int other;           // Opponent's choices (0, 1)
  
  // Dynamic Prior Controls
  real prior_bias_mu;
  real<lower=0> prior_bias_sigma;
  real prior_beta_mu;
  real<lower=0> prior_beta_sigma;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real bias; 
  real beta; 
}

transformed parameters {
  vector[N_trials] memory;
  memory[1] = 0.5; // Neutral start
  
  for (t in 2:N_trials) {
    // Recursive update: running average of 'other' choices
    memory[t] = memory[t-1] + ((other[t-1] - memory[t-1]) / (t-1.0));
    
    // Boundary clipping for numerical stability of the logit link
    memory[t] = fmax(0.01, fmin(0.99, memory[t]));
  }
}

model {
  // Principled regularizing priors
  target += normal_lpdf(bias | prior_bias_mu, prior_bias_sigma);
  target += normal_lpdf(beta | prior_beta_mu, prior_beta_sigma);

  // Likelihood function
  // We use the vectorized bernoulli_logit for computational efficiency
  target += bernoulli_logit_lpmf(y | bias + beta * logit(memory));
}

generated quantities {
  // --- Prior Predictive Checks ---
  real bias_prior = normal_rng(prior_bias_mu, prior_bias_sigma);
  real beta_prior = normal_rng(prior_beta_mu, prior_beta_sigma);
  
  // --- Conditional Diagnostics ---
  array[N_trials] int y_prior_rep;
  int prior_sum;
  array[N_trials] int y_post_rep;
  int post_sum;
  vector[N_trials] log_lik;

  if (run_diagnostics) {
    for (t in 1:N_trials) {
      y_prior_rep[t] = bernoulli_logit_rng(bias_prior + beta_prior * logit(memory[t]));
    }
    // Summary Statistic: Cumulative Choice Rate
    prior_sum = sum(y_prior_rep);

    // --- Posterior Predictive Checks ---
    for (t in 1:N_trials) {
      y_post_rep[t] = bernoulli_logit_rng(bias + beta * logit(memory[t]));
    }
    post_sum = sum(y_post_rep);
    
    for (t in 1:N_trials) {
      log_lik[t] = bernoulli_logit_lpmf(y[t] | bias + beta * logit(memory[t]));
    }
  }

  // 2. Total Log-Prior (for Sensitivity/Update analysis)
  real lprior = normal_lpdf(bias | prior_bias_mu, prior_bias_sigma) + 
                normal_lpdf(beta | prior_beta_mu, prior_beta_sigma);
}

