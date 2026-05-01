
data {
  int<lower=1>                   N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0>          k_self;
  array[N] int<lower=0>          n_self;
  array[N] int<lower=0>          k_social;
  array[N] int<lower=0>          n_social;
  real<lower=0>                  prior_p_alpha;
  real<lower=0>                  prior_p_beta;
  int<lower=0, upper=1>          run_diagnostics;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  target += beta_lpdf(p | prior_p_alpha, prior_p_beta);
  vector[N] a = 0.5 + p * to_vector(k_self)
                    + (1.0 - p) * to_vector(k_social);
  vector[N] b = 0.5 + p * (to_vector(n_self) - to_vector(k_self))
                    + (1.0 - p) * (to_vector(n_social) - to_vector(k_social));
  target += beta_binomial_lpmf(y | 1, a, b);
}
generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;
  if (run_diagnostics) {
    for (i in 1:N) {
      real a_i = 0.5 + p * k_self[i] + (1.0 - p) * k_social[i];
      real b_i = 0.5 + p * (n_self[i] - k_self[i])
                     + (1.0 - p) * (n_social[i] - k_social[i]);
      log_lik[i] = beta_binomial_lpmf(y[i] | 1, a_i, b_i);
      y_rep[i]   = beta_binomial_rng(1, a_i, b_i);
    }
  }
}

