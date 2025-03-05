
// Memory-based choice model with prior and posterior predictions

data {
 int<lower=1> n;
 array[n] int h;
 array[n] int other;
}

parameters {
  real bias;
  real beta;
}

transformed parameters {
  vector[n] memory;
  
  for (trial in 1:n) {
    if (trial == 1) {
      memory[trial] = 0.5;
    } 
    if (trial < n) {
      memory[trial + 1] = memory[trial] + ((other[trial] - memory[trial]) / (trial + 1));
      if (memory[trial + 1] == 0) { memory[trial + 1] = 0.01; }
      if (memory[trial + 1] == 1) { memory[trial + 1] = 0.99; }
    }
  }
}

model {
  // Priors
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);
  
  // Likelihood
  for (trial in 1:n) {
    target += bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
  }
}

generated quantities {
  // Generate prior samples
  real bias_prior = normal_rng(0, .3);
  real beta_prior = normal_rng(0, .5);
  
  // Variables for predictions
  array[n] int prior_preds;
  array[n] int posterior_preds;
  vector[n] memory_prior;
  vector[n] log_lik;
  
  // Generate predictions at different memory levels
  array[3] real memory_levels = {0.2, 0.5, 0.8}; // Low, neutral, and high memory
  array[3] int prior_preds_memory;
  array[3] int posterior_preds_memory;
  
  // Generate predictions from prior for each memory level
  for (i in 1:3) {
    real logit_memory = logit(memory_levels[i]);
    prior_preds_memory[i] = bernoulli_logit_rng(bias_prior + beta_prior * logit_memory);
    posterior_preds_memory[i] = bernoulli_logit_rng(bias + beta * logit_memory);
  }
  
  // Generate predictions from prior
  memory_prior[1] = 0.5;
  for (trial in 1:n) {
    if (trial == 1) {
      prior_preds[trial] = bernoulli_logit_rng(bias_prior + beta_prior * logit(memory_prior[trial]));
    } else {
      memory_prior[trial] = memory_prior[trial-1] + ((other[trial-1] - memory_prior[trial-1]) / trial);
      if (memory_prior[trial] == 0) { memory_prior[trial] = 0.01; }
      if (memory_prior[trial] == 1) { memory_prior[trial] = 0.99; }
      prior_preds[trial] = bernoulli_logit_rng(bias_prior + beta_prior * logit(memory_prior[trial]));
    }
  }
  
  // Generate predictions from posterior
  for (trial in 1:n) {
    posterior_preds[trial] = bernoulli_logit_rng(bias + beta * logit(memory[trial]));
    log_lik[trial] = bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
  }
}


