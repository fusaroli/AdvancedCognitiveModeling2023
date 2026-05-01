
data {
  int<lower=1>                   N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0>          k_self;
  array[N] int<lower=0>          n_self;
  array[N] int<lower=0>          k_social;
  array[N] int<lower=0>          n_social;
  int<lower=0, upper=1>          run_diagnostics;
}
transformed data {
  real a0 = 0.5;
  real b0 = 0.5;
  array[N] real lo_self;
  array[N] real lo_social;
  for (i in 1:N) {
    lo_self[i]   = log((a0 + k_self[i])   / (b0 + n_self[i]   - k_self[i]));
    lo_social[i] = log((a0 + k_social[i]) / (b0 + n_social[i] - k_social[i]));
  }
}
model {
  for (i in 1:N)
    target += bernoulli_logit_lpmf(y[i] | lo_self[i] + lo_social[i]);
}
generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;
  if (run_diagnostics) {
    for (i in 1:N) {
      real L = lo_self[i] + lo_social[i];
      log_lik[i] = bernoulli_logit_lpmf(y[i] | L);
      y_rep[i]   = bernoulli_logit_rng(L);
    }
  }
}

