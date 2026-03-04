
// W5_MemoryBernoulli_GQ.stan
// Social Memory Agent: Estimates bias and memory sensitivity.

data {
  int<lower=1> n;               // Number of trials
  array[n] int h;               // Agent's choices (0, 1)
  array[n] int other;           // Opponent's choices (0, 1)
  
  // Dynamic Prior Controls
  real prior_bias_m;
  real prior_bias_s;
  real prior_beta_m;
  real prior_beta_s;
}

parameters {
  real bias; 
  real beta; 
}

transformed parameters {
  vector[n] memory;
  memory[1] = 0.5; // Neutral start
  
  for (t in 2:n) {
    // Recursive update: running average of 'other' choices
    memory[t] = memory[t-1] + ((other[t-1] - memory[t-1]) / (t-1.0));
    
    // Boundary clipping for numerical stability of the logit link
    memory[t] = fmax(0.01, fmin(0.99, memory[t]));
  }
}

model {
  // Principled regularizing priors
  target += normal_lpdf(bias | prior_bias_m, prior_bias_s);
  target += normal_lpdf(beta | prior_beta_m, prior_beta_s);

  // Likelihood function
  // We use the vectorized bernoulli_logit for computational efficiency
  target += bernoulli_logit_lpmf(h | bias + beta * logit(memory));
}

generated quantities {
  // --- Prior Predictive Checks ---
  real bias_prior = normal_rng(prior_bias_m, prior_bias_s);
  real beta_prior = normal_rng(prior_beta_m, prior_beta_s);
  
  array[n] int h_prior_rep;
  for (t in 1:n) {
    h_prior_rep[t] = bernoulli_logit_rng(bias_prior + beta_prior * logit(memory[t]));
  }
  
  // Summary Statistic: Cumulative Choice Rate
  int prior_sum = sum(h_prior_rep);

  // --- Posterior Predictive Checks ---
  array[n] int h_post_rep;
  for (t in 1:n) {
    h_post_rep[t] = bernoulli_logit_rng(bias + beta * logit(memory[t]));
  }
  
  int post_sum = sum(h_post_rep);
  
  vector[n] log_lik;
  for (t in 1:n) {
    log_lik[t] = bernoulli_logit_lpmf(h[t] | bias + beta * logit(memory[t]));
  }

  // 2. Total Log-Prior (for Sensitivity/Update analysis)
  real lprior = normal_lpdf(bias | prior_bias_m, prior_bias_s) + 
                normal_lpdf(beta | prior_beta_m, prior_beta_s);
}

