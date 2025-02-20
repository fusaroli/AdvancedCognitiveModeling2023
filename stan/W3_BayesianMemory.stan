
data {
  int<lower=1> n;  // number of trials
  array[n] int h;  // agent's choices (0 or 1)
  array[n] int other;  // other player's choices (0 or 1)
}

parameters {
  real<lower=0> alpha_prior;  // Prior alpha parameter
  real<lower=0> beta_prior;   // Prior beta parameter
}

transformed parameters {
  vector[n] alpha;  // Alpha parameter at each trial
  vector[n] beta;   // Beta parameter at each trial
  vector[n] rate;   // Expected rate at each trial
  
  // Initialize with prior
  alpha[1] = alpha_prior;
  beta[1] = beta_prior;
  rate[1] = alpha[1] / (alpha[1] + beta[1]);
  
  // Sequential updating of Beta distribution
  for(t in 2:n) {
    // Update Beta parameters based on previous observation
    alpha[t] = alpha[t-1] + other[t-1];
    beta[t] = beta[t-1] + (1 - other[t-1]);
    
    // Calculate expected rate
    rate[t] = alpha[t] / (alpha[t] + beta[t]);
  }
}

model {
  // Priors on hyperparameters
  target += gamma_lpdf(alpha_prior | 2, 1);
  target += gamma_lpdf(beta_prior | 2, 1);
  
  // Agent's choices follow current rate estimates
  for(t in 1:n) {
    target += bernoulli_lpmf(h[t] | rate[t]);
  }
}

generated quantities {
  array[n] int prior_preds;
  array[n] int posterior_preds;
  real initial_rate = alpha_prior / (alpha_prior + beta_prior);
  
  // Prior predictions use initial rate
  for(t in 1:n) {
    prior_preds[t] = bernoulli_rng(initial_rate);
  }
  
  // Posterior predictions use sequentially updated rates
  for(t in 1:n) {
    posterior_preds[t] = bernoulli_rng(rate[t]);
  }
}

