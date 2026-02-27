
// Memory-based choice model with prior and posterior predictions

data {
 int<lower=1> n;
 array[n] int h;
 array[n] int other;
}

transformed data {
  // WORKFLOW RULE: If a variable depends ONLY on observed data (like 'other'), 
  // calculate it in 'transformed data'. It evaluates exactly ONCE.
  // If you put this in 'transformed parameters', Stan would needlessly recalculate 
  // the exact same vector at every single MCMC leapfrog step, drastically slowing down your model.
  vector[n] memory;
  
  memory[1] = 0.5;
  
  for (trial in 1:(n-1)) {
     memory[trial + 1] = memory[trial] + ((other[trial] - memory[trial]) / (trial + 1));
     // Numerical stability clips
     if (memory[trial + 1] < 0.01) { memory[trial + 1] = 0.01; }
     if (memory[trial + 1] > 0.99) { memory[trial + 1] = 0.99; }
  }
}

parameters {
  real bias;
  real beta;
}

model {
  // Priors
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);
  
  // Likelihood
  // A trial by trial version would be
  // for (trial in 1:n) {
  //   target += bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
  // }
  // However, we vectorize the likelihood for speed.
  target += bernoulli_logit_lpmf(h | bias + beta * logit(memory));
}

generated quantities {
  // We save the pointwise log-likelihood for later model comparisons
  vector[n] log_lik;
  for (trial in 1:n) {
    log_lik[trial] = bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
  }
  
  // Generate prior samples for Prior-Posterior Update checks
  real bias_prior = normal_rng(0, 0.3);
  real beta_prior = normal_rng(0, 0.5);
}

