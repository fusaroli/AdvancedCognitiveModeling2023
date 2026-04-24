
// Memory-based choice model with prior and posterior predictions.
// memory[] is data-derived and lives in transformed data (computed once).
// run_diagnostics gates expensive generated quantities for SBC speed.

data {
 int<lower=1> n;
 array[n] int h;
 array[n] int other;
 int<lower=0, upper=1> run_diagnostics; // 1 = compute PPC + log_lik; 0 = skip (use for SBC)
}

transformed data {
  // memory depends only on other[] (data), so it runs ONCE here.
  // Moving this to transformed parameters would recompute it ~24 000 times per fit.
  vector[n] memory;
  memory[1] = 0.5;
  for (trial in 1:(n-1)) {
     memory[trial + 1] = memory[trial] + ((other[trial] - memory[trial]) / (trial + 1));
     if (memory[trial + 1] < 0.01) memory[trial + 1] = 0.01;
     if (memory[trial + 1] > 0.99) memory[trial + 1] = 0.99;
  }
}

parameters {
  real bias;
  real beta;
}

model {
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);
  // Vectorized likelihood — no loop needed because memory is precomputed.
  target += bernoulli_logit_lpmf(h | bias + beta * logit(memory));
}

generated quantities {
  // Prior samples are always generated (cheap).
  real bias_prior = normal_rng(0, 0.3);
  real beta_prior = normal_rng(0, 0.5);
  // Expensive diagnostics are gated — set run_diagnostics = 0 during SBC.
  vector[n] log_lik;
  array[n] int y_rep;
  if (run_diagnostics) {
    for (trial in 1:n) {
      log_lik[trial] = bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
      y_rep[trial]   = bernoulli_logit_rng(bias + beta * logit(memory[trial]));
    }
  }
}

