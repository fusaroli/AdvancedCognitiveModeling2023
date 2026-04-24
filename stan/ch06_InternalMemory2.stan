
// Memory model with exponential forgetting (learning-rate style).
// memory[] depends on the free parameter 'forgetting', so it cannot go in
// transformed data. It lives as a local vector inside the model block —
// computed at every leapfrog step but never saved to the posterior draws.

data {
  int<lower=1> n;
  array[n] int h;
  array[n] int other;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real bias;
  real beta;
  real<lower=0, upper=1> forgetting;
}

model {
  target += beta_lpdf(forgetting | 1, 1);
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);

  // memory is a LOCAL variable — declared inside model, so it lives on the
  // stack and is discarded after each leapfrog step without touching the saved-
  // draw tape. This is the right pattern when memory depends on a parameter.
  vector[n] memory;
  for (trial in 1:n) {
    if (trial == 1) memory[trial] = 0.5;
    target += bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
    if (trial < n) {
      memory[trial + 1] = (1 - forgetting) * memory[trial] + forgetting * other[trial];
      if (memory[trial + 1] <= 0) memory[trial + 1] = 0.01;
      if (memory[trial + 1] >= 1) memory[trial + 1] = 0.99;
    }
  }
}

generated quantities {
  real bias_prior      = normal_rng(0, 0.3);
  real beta_prior      = normal_rng(0, 0.5);
  real forgetting_prior = beta_rng(1, 1);
  vector[n] log_lik;
  array[n] int y_rep;
  if (run_diagnostics) {
    // Recompute memory for the saved draw — runs once per posterior sample.
    vector[n] memory_gq;
    for (trial in 1:n) {
      if (trial == 1) memory_gq[trial] = 0.5;
      real lin = bias + beta * logit(memory_gq[trial]);
      log_lik[trial] = bernoulli_logit_lpmf(h[trial] | lin);
      y_rep[trial]   = bernoulli_logit_rng(lin);
      if (trial < n) {
        memory_gq[trial + 1] = (1 - forgetting) * memory_gq[trial] + forgetting * other[trial];
        if (memory_gq[trial + 1] <= 0) memory_gq[trial + 1] = 0.01;
        if (memory_gq[trial + 1] >= 1) memory_gq[trial + 1] = 0.99;
      }
    }
  }
}

