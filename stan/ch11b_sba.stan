
data {
  int<lower=1>                   N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0>          k_self;
  array[N] int<lower=0>          n_self;
  array[N] int<lower=0>          k_social;
  array[N] int<lower=0>          n_social;
  int<lower=0, upper=1>          run_diagnostics;
}
model {
  vector[N] a = 0.5 + to_vector(k_self) + to_vector(k_social);
  vector[N] b = 0.5 + (to_vector(n_self)   - to_vector(k_self))
                    + (to_vector(n_social) - to_vector(k_social));
  target += beta_binomial_lpmf(y | 1, a, b);
}
generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;
  if (run_diagnostics) {
    for (i in 1:N) {
      real a_i = 0.5 + k_self[i] + k_social[i];
      real b_i = 0.5 + (n_self[i] - k_self[i]) + (n_social[i] - k_social[i]);
      log_lik[i] = beta_binomial_lpmf(y[i] | 1, a_i, b_i);
      y_rep[i]   = beta_binomial_rng(1, a_i, b_i);
    }
  }
}

